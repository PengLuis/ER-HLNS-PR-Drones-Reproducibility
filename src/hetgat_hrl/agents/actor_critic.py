from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Normal

from hetgat_hrl.agents.task_attention import TaskAttentionModule
from hetgat_hrl.core.algorithm_profile import er_hlns_route_plan_active
from hetgat_hrl.core.mdp_spec import AgentKind, TaskKind, TaskStatus, TruckAction, UAVAction


from hetgat_hrl.core.runtime_constants import DEPOT_DOCK_ID


def _is_uav_delivery_task(task) -> bool:
    """V2 keeps BULK_RELAY tasks NORMAL while allowing UAV execution."""
    if task is None:
        return False
    relay = bool(
        task.kind == TaskKind.NORMAL
        and str(getattr(task, "service_mode", "DIRECT")).strip().upper()
        == "BULK_RELAY"
    )
    return bool(task.kind == TaskKind.EMERGENCY or relay)


class SimpleActorCritic(nn.Module):
    """
    Lightweight actor-critic stub for rebuild stage.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: Tensor) -> Dict[str, Tensor]:
        h = self.encoder(obs)
        return {"logits": self.actor(h), "value": self.critic(h).squeeze(-1)}


class RuleBasedLowLevelPolicy:
    """
    Baseline executable policy that follows assigned tasks.
    """
    MODE_DOCKED_STABLE = "DOCKED_STABLE"
    MODE_RECOVERY_APPROACH = "RECOVERY_APPROACH"
    MODE_READY_FOR_SORTIE = "READY_FOR_SORTIE"
    MODE_SORTIE_EXECUTION = "SORTIE_EXECUTION"
    MODE_SAFE_HOLD = "SAFE_HOLD"

    def __init__(
        self,
        seed: int = 0,
        uav_bind_cooldown_steps: int = 2,
        uav_takeoff_cooldown_steps: int = 2,
        rth_safety_factor: Optional[float] = None,
        service_battery_buffer: float = 0.02,
        short_sortie_max_distance_m: float = 900.0,
        short_sortie_min_battery: float = 0.40,
        stable_goal_before_takeoff_steps: int = 4,
        docked_stability_steps: int = 2,
        idle_recovery_battery_threshold: float = 0.65,
    ):
        self.rng = np.random.default_rng(seed)
        self.uav_bind_cooldown_steps = int(max(uav_bind_cooldown_steps, 0))
        self.uav_takeoff_cooldown_steps = int(max(uav_takeoff_cooldown_steps, 0))
        self.rth_safety_factor = None if rth_safety_factor is None else float(max(rth_safety_factor, 0.0))
        self.service_battery_buffer = float(max(service_battery_buffer, 0.0))
        self.short_sortie_max_distance_m = float(max(short_sortie_max_distance_m, 1.0))
        self.short_sortie_min_battery = float(np.clip(short_sortie_min_battery, 0.0, 1.0))
        self.stable_goal_before_takeoff_steps = int(max(stable_goal_before_takeoff_steps, 0))
        self.docked_stability_steps = int(max(docked_stability_steps, 0))
        self.idle_recovery_battery_threshold = float(np.clip(idle_recovery_battery_threshold, 0.0, 1.0))

        self._last_bind_step: Dict[str, int] = {}
        self._last_takeoff_step: Dict[str, int] = {}
        self._last_goal_id: Dict[str, Optional[str]] = {}
        self._prev_goal_id: Dict[str, Optional[str]] = {}
        self._goal_assigned_step: Dict[str, int] = {}
        self._last_opportunistic_switch_step: Dict[str, int] = {}
        self._uav_mode: Dict[str, str] = {}
        self._uav_mode_assigned_step: Dict[str, int] = {}
        self._docked_steps: Dict[str, int] = {}
        self._last_follow_target_observed: Dict[str, Optional[str]] = {}
        self._takeoff_cmd_latch: Dict[str, bool] = {}
        self._last_seen_step: int = -1

        # Episode diagnostics consumed by rolling eval/calibration scripts.
        self.bind_count_episode: int = 0
        self.takeoff_count_episode: int = 0
        self.docked_stable_steps_episode: int = 0
        self.safe_hold_steps_episode: int = 0

    def _episode_reset_if_needed(self, env) -> None:
        step_now = int(env.state.step_index)
        if (step_now == 0 and self._last_seen_step > 0) or step_now < self._last_seen_step:
            self._last_bind_step = {}
            self._last_takeoff_step = {}
            self._last_goal_id = {}
            self._prev_goal_id = {}
            self._goal_assigned_step = {}
            self._last_opportunistic_switch_step = {}
            self._uav_mode = {}
            self._uav_mode_assigned_step = {}
            self._docked_steps = {}
            self._last_follow_target_observed = {}
            self._takeoff_cmd_latch = {}
            self.bind_count_episode = 0
            self.takeoff_count_episode = 0
            self.docked_stable_steps_episode = 0
            self.safe_hold_steps_episode = 0
        self._last_seen_step = step_now

    def _set_uav_mode(self, aid: str, mode: str, step_now: int) -> None:
        prev = self._uav_mode.get(str(aid), None)
        if prev != mode:
            self._uav_mode_assigned_step[str(aid)] = int(step_now)
        elif str(aid) not in self._uav_mode_assigned_step:
            self._uav_mode_assigned_step[str(aid)] = int(step_now)
        self._uav_mode[str(aid)] = str(mode)

    def _get_uav_mode(self, aid: str) -> str:
        return str(self._uav_mode.get(str(aid), self.MODE_SAFE_HOLD))

    @staticmethod
    def _agent_xy(env, aid: str) -> Tuple[float, float]:
        st = env.state.agents[aid]
        if st.pos_xy is not None:
            return float(st.pos_xy[0]), float(st.pos_xy[1])
        return env._node_xy(int(st.node or 0))

    @staticmethod
    def _full_speed_to(src: Tuple[float, float], dst: Tuple[float, float], vmax: float) -> Tuple[float, float]:
        dx = float(dst[0] - src[0])
        dy = float(dst[1] - src[1])
        norm = float(np.hypot(dx, dy))
        if norm <= 1e-6:
            return 0.0, 0.0
        return float(dx / norm * vmax), float(dy / norm * vmax)

    def _required_rth_battery(self, env, aid: str, dist_to_truck: float) -> float:
        s = env.state.agents[aid]
        base_discharge_per_m = float(max(getattr(env.cfg, "uav_flight_discharge_per_m", 1e-6), 1e-6))
        headwind_coeff = float(max(getattr(env.cfg, "uav_headwind_energy_coeff", 0.04), 0.0))
        rain_coeff = float(max(getattr(env.cfg, "uav_rain_energy_coeff", 0.02), 0.0))
        base_wind = float(max(getattr(env.cfg, "base_wind_mps", 0.0), 0.0))
        base_rain = float(max(getattr(env.cfg, "base_rainfall_mmh", 0.0), 0.0))
        cargo_unit_kg = float(max(getattr(env.cfg, "cargo_unit_kg", 200.0), 1e-6))
        m_load_kg = float(max(getattr(s, "cargo", 0.0), 0.0)) * cargo_unit_kg
        load_factor = 1.0 + 0.018 * m_load_kg
        weather_factor = 1.0 + headwind_coeff * base_wind + rain_coeff * base_rain
        safe_discharge_rate = base_discharge_per_m * weather_factor * load_factor
        return float(max(0.0, dist_to_truck) * safe_discharge_rate)

    def _nearest_truck_from_xy(
        self,
        env,
        xy: Tuple[float, float],
        require_emergency_stock: bool = False,
    ) -> Tuple[Optional[str], float]:
        x, y = float(xy[0]), float(xy[1])
        best_id: Optional[str] = None
        best_d = float("inf")
        for tid, ts in env.state.agents.items():
            if ts.kind != AgentKind.TRUCK or bool(getattr(ts, "crashed", False)):
                continue
            if require_emergency_stock and int(getattr(ts, "emergency_supply_units", 0)) <= 0:
                continue
            txy = ts.pos_xy if ts.pos_xy is not None else env._node_xy(int(ts.node or 0))
            tx, ty = float(txy[0]), float(txy[1])
            d = float(np.hypot(x - tx, y - ty))
            if d < best_d:
                best_d = d
                best_id = str(tid)
        return best_id, best_d

    def _uav_recovery_direction_hint(self, env, aid: str, goal_id: Optional[str]) -> Optional[Tuple[float, float]]:
        cur_xy = self._agent_xy(env, str(aid))

        task = env.state.tasks.get(str(goal_id), None) if goal_id is not None else None
        if task is not None and task.kind == TaskKind.EMERGENCY and task.status.name == "PENDING":
            txy = env._node_xy(int(task.demand_node))
            vx = float(txy[0] - cur_xy[0])
            vy = float(txy[1] - cur_xy[1])
            norm = float(np.hypot(vx, vy))
            if norm > 1e-6:
                return (float(vx / norm), float(vy / norm))

        best_task = None
        best_dist = float("inf")
        for t in env.state.tasks.values():
            if t.kind != TaskKind.EMERGENCY or t.status.name != "PENDING":
                continue
            d = float(env._agent_distance_to_task(str(aid), t))
            if np.isfinite(d) and d < best_dist:
                best_dist = float(d)
                best_task = t
        if best_task is not None:
            txy = env._node_xy(int(best_task.demand_node))
            vx = float(txy[0] - cur_xy[0])
            vy = float(txy[1] - cur_xy[1])
            norm = float(np.hypot(vx, vy))
            if norm > 1e-6:
                return (float(vx / norm), float(vy / norm))

        a = env.state.agents.get(str(aid), None)
        if a is not None and a.vel_xy is not None:
            vx = float(a.vel_xy[0])
            vy = float(a.vel_xy[1])
            norm = float(np.hypot(vx, vy))
            if norm > 1e-6:
                return (float(vx / norm), float(vy / norm))

        return None

    def _best_directional_recovery_truck(
        self,
        env,
        aid: str,
        *,
        require_emergency_stock: bool,
        direction_hint: Optional[Tuple[float, float]],
    ) -> Optional[str]:
        cur_xy = self._agent_xy(env, str(aid))
        world_norm = float(max(getattr(env.cfg, "map_size_m", 3000.0), 1.0))
        w_dir = float(max(getattr(env.cfg, "uav_recovery_direction_weight", 0.70), 0.0))
        w_dist = float(max(getattr(env.cfg, "uav_recovery_distance_weight", 0.55), 0.0))
        w_stock = float(max(getattr(env.cfg, "uav_recovery_stock_weight", 0.20), 0.0))
        init_e = float(max(getattr(env.cfg, "truck_initial_emergency_supply_units", 1), 1))

        best_tid: Optional[str] = None
        best_score = -1e9
        for tid, ts in env.state.agents.items():
            if ts.kind != AgentKind.TRUCK or bool(getattr(ts, "crashed", False)):
                continue
            emer_units = float(max(getattr(ts, "emergency_supply_units", 0), 0.0))
            if require_emergency_stock and emer_units <= 0.0:
                continue
            txy = ts.pos_xy if ts.pos_xy is not None else env._node_xy(int(ts.node or 0))
            dx = float(txy[0] - cur_xy[0])
            dy = float(txy[1] - cur_xy[1])
            d = float(np.hypot(dx, dy))
            if not np.isfinite(d):
                continue

            align = 0.0
            if direction_hint is not None and d > 1e-6:
                ux = float(dx / d)
                uy = float(dy / d)
                cos_v = float(np.clip(direction_hint[0] * ux + direction_hint[1] * uy, -1.0, 1.0))
                align = float((cos_v + 1.0) * 0.5)

            dist_score = float(1.0 - np.clip(d / world_norm, 0.0, 1.0))
            stock_score = float(np.clip(emer_units / init_e, 0.0, 1.0))
            follower_ratio = float(
                np.clip(
                    self._followers_count(env, str(tid), exclude_aid=str(aid))
                    / max(int(getattr(env.cfg, "uav_max_followers_per_truck", 1)), 1),
                    0.0,
                    1.0,
                )
            )
            score = float(w_dist * dist_score + w_dir * align + w_stock * stock_score - 0.10 * follower_ratio)

            if score > best_score + 1e-12:
                best_score = float(score)
                best_tid = str(tid)

        return best_tid

    def _uav_needs_reload(self, env, aid: str) -> bool:
        a = env.state.agents[aid]
        return bool(
            bool(getattr(a, "uav_needs_reload_flag", False))
            or int(getattr(a, "carried_emergency_units", 0)) <= 0
            or float(getattr(a, "payload_kg_current", 0.0)) < float(getattr(env.cfg, "emergency_task_demand_kg", 20.0)) - 1e-9
        )

    def _all_trucks_emergency_empty(self, env) -> bool:
        has_truck = False
        for ts in env.state.agents.values():
            if ts.kind != AgentKind.TRUCK or bool(getattr(ts, "crashed", False)):
                continue
            has_truck = True
            if int(getattr(ts, "emergency_supply_units", 0)) > 0:
                return False
        return bool(has_truck)

    def _task_risk_proxy(self, env, task) -> float:
        node = env.topology.nodes[int(task.demand_node)]
        weather = env.hazards.node_weather(int(task.demand_node))
        rain_norm = float(weather.rain / max(getattr(env.cfg, "base_rainfall_mmh", 1.0), 1e-6))
        wind_norm = float(weather.wind / max(getattr(env.cfg, "base_wind_mps", 1.0), 1e-6))
        quake = float(weather.quake)
        slope = float(getattr(node, "slope_norm", 0.0))
        return float(np.clip(0.30 * rain_norm + 0.25 * wind_norm + 0.30 * quake + 0.15 * slope, 0.0, 3.0))

    def _task_hard_blocked(self, env, task) -> bool:
        if task is None or not _is_uav_delivery_task(task):
            return True
        weather = env.hazards.node_weather(int(task.demand_node))
        max_wind = getattr(env.cfg, "max_uav_wind_mps", None)
        max_rain = getattr(env.cfg, "max_uav_rainfall_mmh", None)
        max_risk = getattr(env.cfg, "max_uav_node_risk", None)
        if max_wind is not None and float(weather.wind) > float(max_wind):
            return True
        if max_rain is not None and float(weather.rain) > float(max_rain):
            return True
        if max_risk is not None and self._task_risk_proxy(env, task) > float(max_risk):
            return True
        return False

    def _mission_required_battery(self, env, aid: str, task) -> float:
        if task is None:
            return float("inf")
        d_go = float(env._agent_distance_to_task(aid, task))
        if not np.isfinite(d_go):
            return float("inf")
        n = env.topology.nodes[int(task.demand_node)]
        _, d_back = self._nearest_truck_from_xy(env, (float(n.x), float(n.y)))
        if not np.isfinite(d_back):
            return float("inf")
        recovery_buf = float(max(getattr(env.cfg, "uav_recovery_distance_buffer_m", 600.0), 0.0))
        req = self._required_rth_battery(env, aid, d_go + d_back + recovery_buf)
        safety = (
            float(self.rth_safety_factor)
            if self.rth_safety_factor is not None
            else float(max(getattr(env.cfg, "rth_safety_factor", 1.2), 0.0))
        )
        service_buf = float(max(getattr(env.cfg, "service_battery_buffer", self.service_battery_buffer), 0.0))
        return float((req + service_buf) * safety)

    def _mission_battery_feasible(self, env, aid: str, task) -> bool:
        if task is None or not _is_uav_delivery_task(task) or task.status.name != "PENDING":
            return False
        if self._task_hard_blocked(env, task):
            return False
        req = self._mission_required_battery(env, aid, task)
        if not np.isfinite(req):
            return False
        batt = float(getattr(env.state.agents[aid], "battery", 0.0))
        return bool(batt >= req)

    def _goal_stable_for_takeoff(self, env, aid: str, step_now: int) -> bool:
        hold_need = int(
            max(
                getattr(
                    env.cfg,
                    "stable_goal_before_takeoff_steps",
                    self.stable_goal_before_takeoff_steps,
                ),
                0,
            )
        )
        assigned = self._goal_assigned_step.get(str(aid), None)
        if assigned is None:
            return bool(hold_need <= 0)
        return bool(int(step_now) - int(assigned) >= hold_need)

    def _goal_id_still_valid_for_uav(self, env, goal_id: Optional[str]) -> bool:
        if goal_id is None:
            return False
        gid = str(goal_id)
        ag = env.state.agents.get(gid, None)
        if ag is not None:
            return bool(ag.kind == AgentKind.TRUCK and (not bool(getattr(ag, "crashed", False))))
        task = env.state.tasks.get(gid, None)
        if task is None:
            return False
        return bool(_is_uav_delivery_task(task) and task.status.name == "PENDING")

    def _is_island_goal_id(self, env, goal_id: Optional[str]) -> bool:
        if goal_id is None:
            return False
        gid = str(goal_id)
        task = env.state.tasks.get(gid, None)
        if task is None or task.kind != TaskKind.EMERGENCY or task.status.name != "PENDING":
            return False
        fn = getattr(env, "_current_island_emergency_task_ids", None)
        if not callable(fn):
            return False
        try:
            island_ids = set(fn())
        except Exception:
            return False
        return bool(str(task.task_id) in island_ids)

    def _goal_distance_for_uav(self, env, aid: str, goal_id: Optional[str]) -> float:
        if goal_id is None:
            return float("inf")
        gid = str(goal_id)
        ag = env.state.agents.get(gid, None)
        if ag is not None and ag.kind == AgentKind.TRUCK:
            ax, ay = self._agent_xy(env, str(aid))
            tx, ty = self._agent_xy(env, gid)
            return float(np.hypot(ax - tx, ay - ty))
        task = env.state.tasks.get(gid, None)
        if task is not None and _is_uav_delivery_task(task) and task.status.name == "PENDING":
            return float(env._agent_distance_to_task(str(aid), task))
        return float("inf")

    def _uav_recovery_urgent(self, env, aid: str) -> bool:
        a = env.state.agents[aid]
        latch = bool(getattr(env, "_uav_forced_rth_latch", {}).get(str(aid), False))
        hard_thr = float(
            np.clip(
                getattr(
                    env.cfg,
                    "uav_hard_recovery_battery_threshold",
                    max(getattr(env.cfg, "uav_short_sortie_min_battery", self.short_sortie_min_battery), 0.35),
                ),
                0.0,
                1.0,
            )
        )
        low_batt = bool(float(getattr(a, "battery", 0.0)) <= hard_thr)
        needs_reload = bool(getattr(a, "uav_needs_reload_flag", False))
        low_cargo = bool(float(getattr(a, "cargo", 0.0)) <= 0.0)
        return bool(latch or low_batt or needs_reload or low_cargo)

    def _maybe_docked_opportunistic_goal(
        self,
        env,
        aid: str,
        current_goal: Optional[str],
        step_now: int,
    ) -> Optional[str]:
        enable_global_v2_departure_radius_experiment = False
        if enable_global_v2_departure_radius_experiment and er_hlns_route_plan_active(env):
            return current_goal
        if not bool(getattr(env.cfg, "uav_docked_opportunistic_retarget_enabled", True)):
            return current_goal
        st = env.state.agents.get(str(aid), None)
        if st is None or st.kind != AgentKind.UAV:
            return current_goal
        if st.follow_target is None:
            return current_goal
        if self._uav_recovery_urgent(env, str(aid)):
            return current_goal
        if bool(getattr(st, "uav_needs_reload_flag", False)):
            return current_goal
        if float(getattr(st, "cargo", 0.0)) <= 0.0:
            return current_goal

        cooldown = int(max(getattr(env.cfg, "uav_docked_opportunistic_cooldown_steps", 0), 0))
        last_sw = self._last_opportunistic_switch_step.get(str(aid), None)
        if last_sw is not None and (int(step_now) - int(last_sw)) < cooldown:
            return current_goal

        radius = float(max(getattr(env.cfg, "uav_docked_opportunistic_radius_m", 0.0), 0.0))
        gain_ratio = float(np.clip(getattr(env.cfg, "uav_docked_opportunistic_gain_ratio", 0.55), 0.05, 0.99))
        min_margin = float(max(getattr(env.cfg, "uav_docked_opportunistic_min_margin_m", 0.0), 0.0))

        island_ids = set()
        if hasattr(env, "_current_island_emergency_task_ids"):
            try:
                island_ids = set(env._current_island_emergency_task_ids())
            except Exception:
                island_ids = set()

        best_tid: Optional[str] = None
        best_d = float("inf")
        for task in env.state.tasks.values():
            if task.kind != TaskKind.EMERGENCY or task.status.name != "PENDING":
                continue
            # Keep island commitment for UAVs already assigned to island goals,
            # but do not hard-freeze every docked UAV to island-only choices.
            if island_ids and (current_goal is not None) and str(current_goal) in island_ids and str(task.task_id) not in island_ids:
                continue
            if self._task_hard_blocked(env, task):
                continue
            if not self._mission_battery_feasible(env, str(aid), task):
                continue
            if hasattr(env, "is_task_serviceable_by_agent"):
                try:
                    if not bool(env.is_task_serviceable_by_agent(str(aid), task)):
                        continue
                except Exception:
                    pass
            d = float(env._agent_distance_to_task(str(aid), task))
            if (not np.isfinite(d)) or d > radius:
                continue
            if d < best_d:
                best_d = d
                best_tid = str(task.task_id)

        if best_tid is None:
            return current_goal

        cur_goal = None if current_goal is None else str(current_goal)
        cur_d = float(self._goal_distance_for_uav(env, str(aid), cur_goal))
        cur_valid = bool(cur_goal is not None and self._goal_id_still_valid_for_uav(env, cur_goal))
        cur_goal_agent = env.state.agents.get(cur_goal, None) if cur_goal is not None else None
        cur_goal_is_truck = bool(cur_goal_agent is not None and cur_goal_agent.kind == AgentKind.TRUCK)

        should_switch = False
        if not cur_valid:
            should_switch = True
        elif cur_goal_is_truck:
            should_switch = bool(best_d <= radius)
        elif np.isfinite(cur_d):
            # If current goal is already near enough, keep it to avoid local oscillation.
            if cur_d <= radius:
                should_switch = False
            else:
                # Only allow far->near correction when candidate is significantly better.
                if best_d <= min(radius, cur_d * gain_ratio) and (cur_d - best_d) >= min_margin:
                    should_switch = True

        if should_switch and cur_goal != best_tid:
            prev_prev = self._prev_goal_id.get(str(aid), None)
            aba_window = int(max(getattr(env.cfg, "uav_goal_lock_steps", 0), 0)) * 2
            assigned_step = int(self._goal_assigned_step.get(str(aid), int(step_now)))
            if (
                prev_prev is not None
                and str(best_tid) == str(prev_prev)
                and (int(step_now) - assigned_step) < aba_window
                and cur_goal is not None
                and self._goal_id_still_valid_for_uav(env, str(cur_goal))
                and (not self._uav_recovery_urgent(env, str(aid)))
            ):
                return str(cur_goal)
            self._last_opportunistic_switch_step[str(aid)] = int(step_now)
            return str(best_tid)
        return current_goal

    def _resolve_uav_goal_with_lock(
        self,
        env,
        aid: str,
        proposed_goal: Optional[str],
        step_now: int,
    ) -> Optional[str]:
        st = env.state.agents.get(str(aid), None)
        goal = None if proposed_goal is None else str(proposed_goal)
        if st is None or st.kind != AgentKind.UAV:
            return goal

        prev_goal = self._last_goal_id.get(str(aid), None)
        assigned_step = int(self._goal_assigned_step.get(str(aid), int(step_now)))
        lock_steps = int(max(getattr(env.cfg, "uav_goal_lock_steps", 0), 0))

        if (
            lock_steps > 0
            and prev_goal is not None
            and goal is not None
            and str(goal) != str(prev_goal)
            and (int(step_now) - assigned_step) < lock_steps
            and self._goal_id_still_valid_for_uav(env, str(prev_goal))
            and (not self._uav_recovery_urgent(env, str(aid)))
            and (not self._is_island_goal_id(env, str(goal)))
        ):
            goal = str(prev_goal)

        # Sortie execution lock: when UAV is already in mission execution,
        # keep current emergency goal unless it is no longer safely serviceable.
        prev_mode = self._get_uav_mode(str(aid))
        if (
            prev_mode == self.MODE_SORTIE_EXECUTION
            and prev_goal is not None
            and goal is not None
            and str(goal) != str(prev_goal)
            and (not self._uav_recovery_urgent(env, str(aid)))
        ):
            prev_task = env.state.tasks.get(str(prev_goal), None)
            if (
                prev_task is not None
                and _is_uav_delivery_task(prev_task)
                and prev_task.status.name == "PENDING"
                and (not self._task_hard_blocked(env, prev_task))
                and self._mission_battery_feasible(env, str(aid), prev_task)
            ):
                goal = str(prev_goal)

        # Anti-ABA lock: suppress short-window A->B->A bouncing.
        prev_prev_goal = self._prev_goal_id.get(str(aid), None)
        aba_window = int(max(getattr(env.cfg, "uav_goal_lock_steps", 0), 0)) * 2
        if (
            aba_window > 0
            and prev_goal is not None
            and prev_prev_goal is not None
            and goal is not None
            and str(goal) == str(prev_prev_goal)
            and str(goal) != str(prev_goal)
            and (int(step_now) - assigned_step) < aba_window
            and self._goal_id_still_valid_for_uav(env, str(prev_goal))
            and (not self._uav_recovery_urgent(env, str(aid)))
            and (not self._is_island_goal_id(env, str(goal)))
        ):
            goal = str(prev_goal)

        # Bind-commit lock from environment: during commit window keep truck recovery target.
        commit_tid = getattr(env, "_uav_bind_commit_target", {}).get(str(aid), None)
        commit_until = int(getattr(env, "_uav_bind_commit_until_step", {}).get(str(aid), -1))
        if commit_tid is not None and int(step_now) <= commit_until:
            goal = str(commit_tid)

        # Low-battery hard goal lock:
        # - below force threshold: always keep/route to truck recovery.
        # - below goal-lock threshold: disallow switching to farther goals.
        batt = float(getattr(st, "battery", 0.0))
        force_thr = float(np.clip(getattr(env.cfg, "uav_low_battery_force_recover_threshold", 0.25), 0.0, 1.0))
        lock_thr = float(np.clip(getattr(env.cfg, "uav_low_battery_goal_lock_threshold", 0.35), 0.0, 1.0))
        if batt < force_thr:
            if prev_goal is not None:
                prev_agent = env.state.agents.get(str(prev_goal), None)
                if prev_agent is not None and prev_agent.kind == AgentKind.TRUCK:
                    goal = str(prev_goal)
                else:
                    near_tid, _ = self._nearest_truck_from_xy(env, self._agent_xy(env, str(aid)))
                    goal = None if near_tid is None else str(near_tid)
            else:
                near_tid, _ = self._nearest_truck_from_xy(env, self._agent_xy(env, str(aid)))
                goal = None if near_tid is None else str(near_tid)
        elif (
            batt < lock_thr
            and prev_goal is not None
            and goal is not None
            and str(goal) != str(prev_goal)
            and self._goal_id_still_valid_for_uav(env, str(prev_goal))
        ):
            prev_d = float(self._goal_distance_for_uav(env, str(aid), str(prev_goal)))
            new_d = float(self._goal_distance_for_uav(env, str(aid), str(goal)))
            if np.isfinite(prev_d) and np.isfinite(new_d) and new_d > prev_d + 1e-6:
                goal = str(prev_goal)

        # Docked anti-far-switch gate: when UAV is already riding truck and has
        # a valid emergency task goal, do not switch to a significantly farther
        # emergency task unless recovery urgency forces it.
        if (
            st.follow_target is not None
            and prev_goal is not None
            and goal is not None
            and str(goal) != str(prev_goal)
            and self._goal_id_still_valid_for_uav(env, str(prev_goal))
            and self._goal_id_still_valid_for_uav(env, str(goal))
            and (not self._uav_recovery_urgent(env, str(aid)))
            and (not self._is_island_goal_id(env, str(goal)))
        ):
            prev_d = float(self._goal_distance_for_uav(env, str(aid), str(prev_goal)))
            new_d = float(self._goal_distance_for_uav(env, str(aid), str(goal)))
            margin = float(max(getattr(env.cfg, "uav_docked_opportunistic_min_margin_m", 0.0), 0.0))
            if np.isfinite(prev_d) and np.isfinite(new_d) and new_d > (prev_d + margin):
                goal = str(prev_goal)

        # Docked opportunistic retarget: if UAV is already riding a truck and a
        # much closer emergency is now serviceable, allow local goal correction.
        goal = self._maybe_docked_opportunistic_goal(
            env,
            str(aid),
            current_goal=goal,
            step_now=int(step_now),
        )

        if prev_goal != goal:
            self._goal_assigned_step[str(aid)] = int(step_now)
            self._prev_goal_id[str(aid)] = None if prev_goal is None else str(prev_goal)
        elif str(aid) not in self._goal_assigned_step:
            self._goal_assigned_step[str(aid)] = int(step_now)
        self._last_goal_id[str(aid)] = goal
        return goal

    def _within_short_sortie_envelope(self, env, aid: str, task) -> bool:
        if task is None:
            return False
        dist = float(env._agent_distance_to_task(aid, task))
        short_max = float(max(getattr(env.cfg, "uav_short_sortie_max_distance_m", self.short_sortie_max_distance_m), 1.0))
        if (not np.isfinite(dist)) or dist > short_max:
            return False
        batt = float(getattr(env.state.agents[aid], "battery", 0.0))
        short_min_batt = float(np.clip(getattr(env.cfg, "uav_short_sortie_min_battery", self.short_sortie_min_battery), 0.0, 1.0))
        if batt < short_min_batt:
            return False
        return True

    def _v2_launch_anchor_ready(self, env, aid: str, task) -> bool:
        """Require the layer-1 truck to reach its published launch anchor."""
        if not er_hlns_route_plan_active(env) or task is None:
            return True
        uid = str(aid)
        task_id = str(getattr(task, "task_id", ""))
        state = env.state.agents.get(uid, None)
        if state is None or state.follow_target is None:
            return True
        truck_id = str(state.follow_target)
        launch_node = None
        assists = getattr(env, "_planner_truck_assist_waypoint_by_truck", {})
        assist = assists.get(truck_id, None) if isinstance(assists, dict) else None
        if (
            isinstance(assist, dict)
            and str(assist.get("uav_id", "")) == uid
            and str(assist.get("task_id", "")) == task_id
        ):
            launch_node = assist.get("launch_node", None)
        if launch_node is None:
            audit = getattr(env, "_planner_route_plan_v2", {})
            routes = audit.get("routes", {}) if isinstance(audit, dict) else {}
            route = routes.get(truck_id, None) if isinstance(routes, dict) else None
            if (
                isinstance(route, dict)
                and str(route.get("current_task_id", "")) == task_id
            ):
                for stop in route.get("stops", ()):
                    if not isinstance(stop, dict) or str(stop.get("task_id", "")) != task_id:
                        continue
                    launch_node = stop.get("selected_anchor", None)
                    if launch_node is None:
                        launch_node = stop.get("target_node", None)
                    break
        if launch_node is None:
            return True
        truck = env.state.agents.get(truck_id, None)
        at_anchor = bool(
            truck is not None
            and truck.node is not None
            and truck.transit is None
            and int(truck.node) == int(launch_node)
        )
        if at_anchor:
            return True
        # The exact anchor is the preferred road stop, not a reason to miss an
        # urgent delivery after the truck has already entered the nominal safe
        # one-way envelope.  The launch gate below still validates battery,
        # weather and recovery; this clause only prevents the old 4--5 km
        # premature departures while retaining safe opportunistic launch.
        try:
            distance = float(env._agent_distance_to_task(uid, task))
        except Exception:
            distance = float("inf")
        safe_departure_radius = float(
            0.50 * max(getattr(env.cfg, "uav_max_sortie_m", 6000.0), 1.0)
        )
        if np.isfinite(distance) and distance <= safe_departure_radius + 1e-9:
            return True
        stay_reasons = getattr(
            env, "_planner_route_plan_stay_reason_by_agent", {}
        )
        anchor_unreachable = bool(
            isinstance(stay_reasons, dict)
            and str(stay_reasons.get(truck_id, ""))
            == "v2_launch_anchor_unreachable"
        )
        if anchor_unreachable and hasattr(env, "_uav_launch_gate_check"):
            try:
                launch_ok, launch_reason, _ = env._uav_launch_gate_check(
                    uid, task=task, count_reject=False
                )
            except TypeError:
                launch_ok, launch_reason, _ = env._uav_launch_gate_check(
                    uid, task=task
                )
            except Exception:
                return False
            return bool(
                launch_ok
                and (
                    str(launch_reason).startswith("direct_safe")
                    or str(launch_reason).startswith("rendezvous_safe")
                )
            )
        return False

    def _allow_takeoff_for_task_goal(self, env, aid: str, task, step_now: int) -> bool:
        if task is None or not _is_uav_delivery_task(task) or task.status.name != "PENDING":
            return False
        if er_hlns_route_plan_active(env):
            try:
                contract_distance = float(env._agent_distance_to_task(str(aid), task))
            except Exception:
                contract_distance = float("inf")
            contract_departure_radius = float(
                (2.0 / 3.0)
                * max(getattr(env.cfg, "uav_max_sortie_m", 6000.0), 1.0)
            )
            state = env.state.agents.get(str(aid), None)
            depot_fallback_safe = False
            anchor_unreachable_safe = False
            # If a road event invalidates the selected launch anchor, do not
            # strand a loaded UAV solely because the nominal anchor is no
            # longer reachable. The environment launch gate remains
            # authoritative for energy, recovery and safety.
            if (
                bool(
                    getattr(
                        env.cfg,
                        "hrl_b_anchor_unreachable_uav_launch_enabled",
                        False,
                    )
                )
                and state is not None
                and getattr(state, "follow_target", None) is not None
                and float(
                    getattr(task, "lifeline_current", 0.0)
                ) / max(float(getattr(task, "lifeline_init", 100.0)), 1e-9)
                <= float(
                    getattr(
                        env.cfg,
                        "hrl_b_anchor_unreachable_uav_launch_max_lifeline_ratio",
                        0.10,
                    )
                )
                and isinstance(
                    getattr(env, "_planner_route_plan_stay_reason_by_agent", {}),
                    dict,
                )
                and str(
                    getattr(env, "_planner_route_plan_stay_reason_by_agent", {}).get(
                        str(getattr(state, "follow_target", "")), ""
                    )
                )
                == "v2_launch_anchor_unreachable"
                and hasattr(env, "_uav_launch_gate_check")
            ):
                try:
                    launch_ok, launch_reason, _ = env._uav_launch_gate_check(
                        str(aid), task=task, count_reject=False
                    )
                    anchor_unreachable_safe = bool(
                        launch_ok
                        and (
                            str(launch_reason).startswith("direct_safe")
                            or str(launch_reason).startswith("rendezvous_safe")
                        )
                    )
                except Exception:
                    anchor_unreachable_safe = False
            truck_emergency_stock_available = True
            if hasattr(env, "_any_truck_with_emergency_stock"):
                try:
                    truck_emergency_stock_available = bool(
                        env._any_truck_with_emergency_stock()
                    )
                except Exception:
                    truck_emergency_stock_available = True
            if (
                state is not None
                and str(getattr(state, "follow_target", "")) == DEPOT_DOCK_ID
                and not truck_emergency_stock_available
                and hasattr(env, "_uav_launch_gate_check")
            ):
                try:
                    launch_ok, launch_reason, _ = env._uav_launch_gate_check(
                        str(aid), task=task
                    )
                    depot_fallback_safe = bool(
                        launch_ok
                        and (
                            str(launch_reason).startswith("direct_safe")
                            or str(launch_reason).startswith("rendezvous_safe")
                        )
                    )
                except Exception:
                    depot_fallback_safe = False
            if (
                not np.isfinite(contract_distance)
                or contract_distance > contract_departure_radius + 1e-9
            ) and not depot_fallback_safe and not anchor_unreachable_safe:
                return False
        if hasattr(env, "_uav_launch_block_cooldown_active"):
            try:
                if bool(env._uav_launch_block_cooldown_active(str(aid))):
                    return False
            except Exception:
                pass
        a = env.state.agents[aid]
        if float(getattr(a, "cargo", 0.0)) <= 0.0:
            return False
        can_rendezvous_corridor = False
        docked_actionable_now = False
        # Docked UAV must finish reload/charge gate before takeoff.
        # This avoids "reload done but no charge" early-depart behavior.
        if a.follow_target is not None:
            min_dwell = int(max(getattr(env.cfg, "uav_post_bind_min_dwell_steps", 0), 0))
            if int(self._docked_steps.get(str(aid), 0)) < min_dwell:
                return False
            if bool(getattr(env.cfg, "uav_post_bind_force_reload", True)) and bool(getattr(a, "uav_needs_reload_flag", False)):
                return False
            if int(getattr(a, "uav_reload_timer", 0)) > 0:
                return False
            if hasattr(env, "_uav_force_takeoff_battery_threshold"):
                try:
                    depart_battery_thr = float(env._uav_force_takeoff_battery_threshold(task=task))
                except TypeError:
                    depart_battery_thr = float(env._uav_force_takeoff_battery_threshold())
            else:
                depart_battery_thr = float(
                    max(getattr(env.cfg, "uav_force_takeoff_battery_threshold", 0.98), 0.0)
                )
            if bool(getattr(env.cfg, "uav_post_bind_force_recharge", True)) and float(getattr(a, "battery", 0.0)) + 1e-9 < depart_battery_thr:
                return False
            enable_exact_v2_anchor_gate_experiment = True
            if (
                enable_exact_v2_anchor_gate_experiment
                and not self._v2_launch_anchor_ready(env, str(aid), task)
            ):
                return False
        # Do not takeoff immediately after a new assignment, unless this is a
        # safety recovery move (which is truck-goal path and does not use takeoff).
        stable_takeoff = bool(self._goal_stable_for_takeoff(env, aid, step_now=step_now))
        forced_task = dict(
            getattr(env, "_planner_force_takeoff_task_by_uav", {}) or {}
        ).get(str(aid), None)
        watchdog_force = bool(
            forced_task is not None
            and str(forced_task) == str(getattr(task, "task_id", ""))
        )
        # Planner watchdog authority is intentionally narrow: it skips only
        # assignment-stability waiting.  Cargo/reload/charge checks above and
        # hard-block, battery, horizon and recovery checks below still apply.
        if watchdog_force:
            stable_takeoff = True
        if (not stable_takeoff) and a.follow_target is not None:
            # Near-goal fast release: if emergency task is already inside short-sortie
            # envelope, avoid over-holding on truck due to assignment hysteresis.
            try:
                d_goal = float(env._agent_distance_to_task(aid, task))
            except Exception:
                d_goal = float('inf')
            near_thr = float(max(getattr(env.cfg, "uav_short_sortie_max_distance_m", self.short_sortie_max_distance_m), 1.0))
            if np.isfinite(d_goal) and d_goal <= near_thr:
                stable_takeoff = True
            elif hasattr(env, "_uav_docked_task_actionable_now"):
                try:
                    docked_actionable_now = bool(env._uav_docked_task_actionable_now(str(aid), task))
                    if docked_actionable_now:
                        stable_takeoff = True
                except Exception:
                    docked_actionable_now = False
        if not stable_takeoff:
            return False
        if self._task_hard_blocked(env, task):
            return False
        if (not docked_actionable_now) and (not self._mission_battery_feasible(env, aid, task)):
            # Island/support corridor: trust environment launch gate when it
            # explicitly allows rendezvous-safe dispatch.
            if hasattr(env, "_uav_launch_gate_check"):
                try:
                    launch_ok, launch_reason, _ = env._uav_launch_gate_check(str(aid), task=task)
                    can_rendezvous_corridor = bool(
                        launch_ok and str(launch_reason).startswith("rendezvous_safe")
                    )
                except Exception:
                    can_rendezvous_corridor = False
            if not can_rendezvous_corridor:
                return False

        # Horizon feasibility gate:
        # avoid late-window takeoff that cannot reasonably complete emergency
        # execution + recovery before episode end.
        max_steps = int(max(getattr(env.cfg, "max_steps", 0), 0))
        rem_steps = int(max(max_steps - int(step_now), 0))
        min_rem = int(max(getattr(env.cfg, "uav_launch_min_remaining_steps", 8), 0))
        if rem_steps < min_rem:
            return False
        d_go = float(env._agent_distance_to_task(aid, task))
        n = env.topology.nodes[int(task.demand_node)]
        _, d_back = self._nearest_truck_from_xy(env, (float(n.x), float(n.y)))
        if np.isfinite(d_go) and np.isfinite(d_back):
            recovery_buf = float(max(getattr(env.cfg, "uav_recovery_distance_buffer_m", 600.0), 0.0))
            if can_rendezvous_corridor:
                bind_window = float(max(getattr(env.cfg, "uav_bind_radius_m", 170.0), 1.0))
                rendez_dist = float(max(0.75 * recovery_buf, bind_window))
                decision_interval = int(max(getattr(env.cfg, "decision_interval", 5), 1))
                dt = float(max(getattr(env.cfg, "dt_seconds", 1.0), 1e-6))
                truck_speed = float(max(getattr(env.cfg, "truck_speed_mps", 0.0), 0.0))
                truck_drift = float(truck_speed * dt * decision_interval)
                # Corridor horizon: island/recovery sortie only needs guaranteed
                # task reach + rendezvous capture, not strict full direct return.
                mission_dist = float(max(d_go + rendez_dist + truck_drift, 0.0))
            else:
                mission_dist = float(max(d_go + d_back + recovery_buf, 0.0))
            util = float(np.clip(getattr(env.cfg, "uav_launch_speed_utilization", 0.70), 0.1, 1.0))
            v_ref = float(max(getattr(env.cfg, "uav_max_speed_mps", 1.0) * util, 1e-6))
            dt = float(max(getattr(env.cfg, "dt_seconds", 1.0), 1e-6))
            est_steps = int(np.ceil(mission_dist / max(v_ref * dt, 1e-6)))
            horizon_buf = int(max(getattr(env.cfg, "uav_launch_min_horizon_buffer_steps", 4), 0))
            if rem_steps < int(est_steps + horizon_buf):
                return False

        # Prefer short safe sorties for departing from truck recovery.
        if docked_actionable_now:
            return True
        if self._within_short_sortie_envelope(env, aid, task):
            return True
        if can_rendezvous_corridor:
            return True
        # Long-range sortie is allowed only with clear battery margin.
        req = self._mission_required_battery(env, aid, task)
        if not np.isfinite(req):
            return False
        batt = float(getattr(env.state.agents[aid], "battery", 0.0))
        return bool(batt >= float(req * 1.10))

    @staticmethod
    def _cooldown_ok(step_now: int, last_step: Optional[int], cooldown: int) -> bool:
        if last_step is None:
            return True
        return bool(int(step_now) - int(last_step) >= int(max(cooldown, 0)))

    def _is_bind_legal(self, env, aid: str, truck_id: str, cur_xy: Tuple[float, float]) -> bool:
        a = env.state.agents[aid]
        if a.follow_target is not None:
            return False
        t = env.state.agents.get(str(truck_id), None)
        if t is None or t.kind != AgentKind.TRUCK or bool(t.crashed):
            return False
        tx, ty = self._agent_xy(env, str(truck_id))
        d = float(np.hypot(cur_xy[0] - tx, cur_xy[1] - ty))
        if d > float(getattr(env.cfg, "uav_bind_radius_m", 50.0)):
            return False
        followers = sum(
            1
            for uid, us in env.state.agents.items()
            if uid != aid and us.kind == AgentKind.UAV and us.follow_target == str(truck_id)
        )
        if followers >= int(max(getattr(env.cfg, "uav_max_followers_per_truck", 0), 0)):
            return False
        return True

    def _followers_count(self, env, truck_id: str, exclude_aid: Optional[str] = None) -> int:
        return int(
            sum(
                1
                for uid, us in env.state.agents.items()
                if us.kind == AgentKind.UAV
                and uid != exclude_aid
                and us.follow_target is not None
                and str(us.follow_target) == str(truck_id)
            )
        )

    def _bind_slot_available(
        self,
        env,
        aid: str,
        truck_id: str,
        bind_reserved: Optional[Dict[str, int]] = None,
    ) -> bool:
        cap = int(max(getattr(env.cfg, "uav_max_followers_per_truck", 0), 0))
        if cap <= 0:
            return False
        already = int(self._followers_count(env, str(truck_id), exclude_aid=str(aid)))
        reserved = int((bind_reserved or {}).get(str(truck_id), 0))
        return bool((already + reserved) < cap)

    def _goal_valid_for_uav(self, goal_id: Optional[str], target_agent, target_task) -> bool:
        if goal_id is None:
            return False
        if target_task is not None:
            return bool(_is_uav_delivery_task(target_task) and target_task.status.name == "PENDING")
        if target_agent is not None:
            return bool(target_agent.kind == AgentKind.TRUCK and (not bool(target_agent.crashed)))
        return False

    def _resolve_recovery_truck(self, env, aid: str, goal_id: Optional[str], target_agent) -> Optional[str]:
        needs_reload = bool(self._uav_needs_reload(env, str(aid)))

        selected_tid: Optional[str] = None
        # A layer-1 cross-truck contract is authoritative over nearest-truck
        # fallback: its receiver has already reserved a follower slot and is
        # moving to the agreed recovery boundary.
        transfer_tid = self._planner_transfer_target_truck(env, str(aid), None)
        if transfer_tid is not None:
            selected_tid = str(transfer_tid)
        if target_agent is not None and target_agent.kind == AgentKind.TRUCK and goal_id is not None:
            if selected_tid is None and ((not needs_reload) or int(getattr(target_agent, "emergency_supply_units", 0)) > 0):
                selected_tid = str(goal_id)

        cur_xy = self._agent_xy(env, aid)
        if selected_tid is None:
            require_stock = bool(needs_reload)
            near_tid, near_dist = self._nearest_truck_from_xy(env, cur_xy, require_emergency_stock=require_stock)
            near_radius = float(max(getattr(env.cfg, "uav_recovery_near_truck_radius_m", 700.0), 0.0))
            no_nearby = bool((near_tid is None) or (not np.isfinite(near_dist)) or (near_dist > near_radius))

            if needs_reload and near_tid is None and bool(self._all_trucks_emergency_empty(env)):
                if bool(getattr(env.cfg, "uav_reload_at_depot_enabled", True)):
                    selected_tid = DEPOT_DOCK_ID

            if selected_tid is None:
                directional_enabled = bool(getattr(env.cfg, "uav_recovery_directional_select_enabled", True))
                if directional_enabled and no_nearby:
                    hint = self._uav_recovery_direction_hint(env, str(aid), goal_id)
                    directional_tid = self._best_directional_recovery_truck(
                        env,
                        str(aid),
                        require_emergency_stock=require_stock,
                        direction_hint=hint,
                    )
                    if directional_tid is not None:
                        selected_tid = str(directional_tid)

            if selected_tid is None and near_tid is not None:
                selected_tid = str(near_tid)

            if selected_tid is None and needs_reload and bool(getattr(env.cfg, "uav_reload_at_depot_enabled", True)):
                selected_tid = DEPOT_DOCK_ID

            if selected_tid is None:
                fallback_tid, _ = self._nearest_truck_from_xy(env, cur_xy, require_emergency_stock=False)
                selected_tid = None if fallback_tid is None else str(fallback_tid)

        # Low-battery recovery request map: exposes UAV-preferred recovery truck
        # to env-side truck support so rendezvous can become a two-way motion.
        request_enabled = bool(getattr(env.cfg, "uav_recovery_truck_request_enabled", True))
        req_map = getattr(env, "_uav_recovery_requested_truck", None)
        if request_enabled and not isinstance(req_map, dict):
            req_map = {}
            setattr(env, "_uav_recovery_requested_truck", req_map)

        batt = float(getattr(env.state.agents[str(aid)], "battery", 0.0))
        low_thr = float(np.clip(getattr(env.cfg, "uav_low_battery_goal_lock_threshold", 0.35), 0.0, 1.0))
        forced_recovery = bool(getattr(env, "_uav_forced_rth_latch", {}).get(str(aid), False))
        urgent_recovery = bool(forced_recovery or batt <= low_thr or needs_reload)

        if isinstance(req_map, dict):
            if urgent_recovery and selected_tid is not None and str(selected_tid) != DEPOT_DOCK_ID:
                req_map[str(aid)] = str(selected_tid)
            else:
                req_map.pop(str(aid), None)

        return selected_tid

    def _planner_transfer_target_truck(self, env, aid: str, task=None) -> Optional[str]:
        req_truck = getattr(env, "_uav_transfer_target_truck", None)
        req_task = getattr(env, "_uav_transfer_target_task", None)
        if not isinstance(req_truck, dict) or not isinstance(req_task, dict):
            return None
        uid = str(aid)
        tid = str(req_truck.get(uid, "")).strip()
        if not tid:
            return None
        ag = env.state.agents.get(str(tid), None)
        if ag is None or ag.kind != AgentKind.TRUCK or bool(getattr(ag, "crashed", False)):
            return None
        hinted_task_id = str(req_task.get(uid, "")).strip()
        if task is not None:
            if not _is_uav_delivery_task(task) or task.status.name != "PENDING":
                return None
            if hinted_task_id and hinted_task_id != str(task.task_id):
                return None
        return str(tid)

    def _safe_recovery_action(
        self,
        env,
        aid: str,
        step_now: int,
        bind_cooldown: int,
        bind_reserved: Optional[Dict[str, int]] = None,
        preferred_truck_id: Optional[str] = None,
    ) -> UAVAction:
        a = env.state.agents[aid]
        if a.follow_target is not None:
            return UAVAction(vx=0.0, vy=0.0)

        cur_xy = self._agent_xy(env, aid)
        target_id = preferred_truck_id
        if target_id is None:
            target_id = self._resolve_recovery_truck(env, str(aid), goal_id=None, target_agent=None)

        if target_id == DEPOT_DOCK_ID:
            depot_xy = env._node_xy(0)
            dist = float(np.hypot(cur_xy[0] - float(depot_xy[0]), cur_xy[1] - float(depot_xy[1])))
            if dist <= float(max(getattr(env.cfg, "uav_bind_radius_m", 50.0), 1.0)):
                return UAVAction(vx=0.0, vy=0.0)
            vmax = float(max(getattr(env.cfg, "uav_max_speed_mps", 1.0), 1e-6))
            vx, vy = self._full_speed_to(cur_xy, (float(depot_xy[0]), float(depot_xy[1])), vmax=vmax)
            return UAVAction(vx=vx, vy=vy)

        near_tid = target_id
        near_dist = float("inf")
        if near_tid is not None:
            tx, ty = self._agent_xy(env, str(near_tid))
            near_dist = float(np.hypot(cur_xy[0] - tx, cur_xy[1] - ty))
        if near_tid is None or (not np.isfinite(near_dist)):
            need_reload = bool(self._uav_needs_reload(env, str(aid)))
            near_tid, near_dist = self._nearest_truck_from_xy(env, cur_xy, require_emergency_stock=need_reload)
            if near_tid is None and need_reload and bool(getattr(env.cfg, "uav_reload_at_depot_enabled", True)):
                depot_xy = env._node_xy(0)
                vmax = float(max(getattr(env.cfg, "uav_max_speed_mps", 1.0), 1e-6))
                vx, vy = self._full_speed_to(cur_xy, (float(depot_xy[0]), float(depot_xy[1])), vmax=vmax)
                return UAVAction(vx=vx, vy=vy)
        if near_tid is None or (not np.isfinite(near_dist)):
            return UAVAction(vx=0.0, vy=0.0)

        # Prevent bind/forced-takeoff ping-pong when UAV is already fully sortie-ready.
        has_open_emergency = bool(getattr(env, "_has_open_emergency_tasks", lambda: False)())
        if hasattr(env, "_uav_force_takeoff_battery_threshold"):
            full_takeoff_thr = float(env._uav_force_takeoff_battery_threshold())
        else:
            full_takeoff_thr = float(max(getattr(env.cfg, "uav_force_takeoff_battery_threshold", 0.95), 0.0))
        cargo_full = bool(
            float(getattr(a, "cargo", 0.0))
            >= float(getattr(env.cfg, "uav_cargo_capacity_units", 1.0))
        )
        dispatch_ready = bool(float(getattr(a, "battery", 0.0)) >= full_takeoff_thr and cargo_full and has_open_emergency)
        if dispatch_ready and near_dist <= float(getattr(env.cfg, "uav_bind_radius_m", 50.0)):
            return UAVAction(vx=0.0, vy=0.0)

        can_bind = self._cooldown_ok(step_now, self._last_bind_step.get(str(aid), None), bind_cooldown)
        if (
            can_bind
            and near_dist <= float(getattr(env.cfg, "uav_bind_radius_m", 50.0))
            and self._is_bind_legal(env, aid, str(near_tid), cur_xy)
            and self._bind_slot_available(env, str(aid), str(near_tid), bind_reserved=bind_reserved)
        ):
            return UAVAction(bind_truck_id=str(near_tid))

        txy = self._agent_xy(env, str(near_tid))
        vmax = float(max(getattr(env.cfg, "uav_max_speed_mps", 1.0), 1e-6))
        vx, vy = self._full_speed_to(cur_xy, txy, vmax=vmax)
        return UAVAction(vx=vx, vy=vy)

    def _decide_uav_mode(
        self,
        env,
        aid: str,
        goal_id: Optional[str],
        target_agent,
        target_task,
        step_now: int,
    ) -> str:
        a = env.state.agents[aid]
        bound_tid = None if a.follow_target is None else str(a.follow_target)
        goal_truck_id = None
        if target_agent is not None and target_agent.kind == AgentKind.TRUCK and goal_id is not None:
            goal_truck_id = str(goal_id)
        has_task_goal = bool(
            target_task is not None and _is_uav_delivery_task(target_task) and target_task.status.name == "PENDING"
        )

        # An accepted V2 sortie is a delivery commitment.  The generic
        # distance-to-nearest-truck recovery latch is intentionally
        # conservative and may fire before a rendezvous mission reaches the
        # task.  Let the authoritative sortie contract finish the delivery
        # leg whenever that leg still fits the actual remaining battery.
        if (
            has_task_goal
            and bound_tid is None
            and self._v2_airborne_delivery_leg_feasible(
                env, str(aid), target_task
            )
        ):
            return self.MODE_SORTIE_EXECUTION

        if self._uav_recovery_urgent(env, aid):
            if bound_tid is not None:
                return self.MODE_DOCKED_STABLE
            return self.MODE_RECOVERY_APPROACH

        if goal_truck_id is not None:
            if bound_tid is not None and bound_tid == goal_truck_id:
                return self.MODE_DOCKED_STABLE
            return self.MODE_RECOVERY_APPROACH

        if has_task_goal:
            if bound_tid is not None:
                transfer_tid = self._planner_transfer_target_truck(env, str(aid), target_task)
                if transfer_tid is not None and str(transfer_tid) != str(bound_tid):
                    try:
                        docked_actionable_now = bool(env._uav_docked_task_actionable_now(str(aid), target_task))
                    except Exception:
                        docked_actionable_now = False
                    if not docked_actionable_now:
                        return self.MODE_RECOVERY_APPROACH
                docked_steps = int(self._docked_steps.get(str(aid), 0))
                ready = bool(
                    self._allow_takeoff_for_task_goal(env, aid, target_task, step_now=step_now)
                    and docked_steps >= int(self.docked_stability_steps)
                )
                return self.MODE_READY_FOR_SORTIE if ready else self.MODE_DOCKED_STABLE
            if self._mission_battery_feasible(env, aid, target_task):
                return self.MODE_SORTIE_EXECUTION
            # A V2 route contract may deliberately use a rendezvous or
            # cross-truck recovery chain.  Its launch gate validates the full
            # delivery + recovery contract, while _mission_battery_feasible
            # above assumes a stricter direct return to a nearby truck.  Do
            # not let that legacy direct-return check turn an accepted sortie
            # into an empty flight immediately after takeoff.  Once airborne,
            # keep the delivery commitment whenever the remaining delivery
            # leg itself is energy-feasible; recovery is executed after the
            # service under the already-published contract.
            if self._v2_airborne_delivery_leg_feasible(env, str(aid), target_task):
                return self.MODE_SORTIE_EXECUTION
            return self.MODE_RECOVERY_APPROACH

        if bound_tid is not None:
            return self.MODE_DOCKED_STABLE

        short_min_batt = float(np.clip(getattr(env.cfg, "uav_short_sortie_min_battery", self.short_sortie_min_battery), 0.0, 1.0))
        low_batt = bool(float(getattr(a, "battery", 0.0)) <= short_min_batt)
        low_cargo = bool(float(getattr(a, "cargo", 0.0)) <= 0.0)
        if low_batt or low_cargo:
            return self.MODE_RECOVERY_APPROACH
        idle_recovery = bool(
            float(getattr(a, "battery", 0.0)) <= float(self.idle_recovery_battery_threshold)
        )
        if idle_recovery:
            return self.MODE_RECOVERY_APPROACH
        return self.MODE_SAFE_HOLD

    def _v2_airborne_delivery_leg_feasible(self, env, aid: str, task) -> bool:
        """Honor an accepted V2 rendezvous sortie through task delivery."""
        state = env.state.agents.get(str(aid), None)
        if (
            state is None
            or state.kind != AgentKind.UAV
            or state.follow_target is not None
            or task is None
            or not _is_uav_delivery_task(task)
            or task.status.name != "PENDING"
        ):
            return False
        route_audit = getattr(env, "_planner_route_plan_v2", {})
        contracts = (
            route_audit.get("contracts", {})
            if isinstance(route_audit, dict)
            else {}
        )
        contract = (
            contracts.get(str(task.task_id), None)
            if isinstance(contracts, dict)
            else None
        )
        sortie_contracts = getattr(env, "_uav_sortie_contract_task", {})
        owns_environment_sortie = bool(
            isinstance(sortie_contracts, dict)
            and str(sortie_contracts.get(str(aid), "")) == str(task.task_id)
        )
        owns_current_plan = bool(
            isinstance(contract, dict)
            and str(contract.get("owner_agent_id", "")) == str(aid)
        )
        if not (owns_environment_sortie or owns_current_plan):
            return False
        try:
            cur_xy = self._agent_xy(env, str(aid))
            d_go = float(env._agent_distance_to_task(str(aid), task))
            required = float(
                env._uav_energy_cost_fraction(str(aid), float(d_go), cur_xy)
            )
            service_buffer = float(
                max(getattr(env.cfg, "service_battery_buffer", 0.0), 0.0)
            )
            battery = float(max(getattr(state, "battery", 0.0), 0.0))
            return bool(
                np.isfinite(d_go)
                and np.isfinite(required)
                and battery + 1e-9 >= required + service_buffer
            )
        except Exception:
            return False

    def _return_feasible_after_action(self, env, aid: str, action: UAVAction) -> bool:
        a = env.state.agents[aid]
        if a.follow_target is not None and not bool(action.takeoff):
            return True
        if action.bind_truck_id is not None:
            return True

        cur_xy = self._agent_xy(env, aid)
        vmax = float(max(getattr(env.cfg, "uav_max_speed_mps", 1.0), 1e-6))
        vx = float(action.vx)
        vy = float(action.vy)
        speed = float(np.hypot(vx, vy))
        if speed > vmax and speed > 1e-6:
            sc = float(vmax / speed)
            vx *= sc
            vy *= sc
        dt = float(max(getattr(env.cfg, "dt_seconds", 1.0), 1e-6))
        world_lim = float(max(getattr(env.cfg, "map_size_m", 3000.0), 1.0))
        nx = float(np.clip(cur_xy[0] + vx * dt, 0.0, world_lim))
        ny = float(np.clip(cur_xy[1] + vy * dt, 0.0, world_lim))
        _, d_back = self._nearest_truck_from_xy(env, (nx, ny))
        if not np.isfinite(d_back):
            return False
        req = self._required_rth_battery(env, aid, float(d_back))
        safety = (
            float(self.rth_safety_factor)
            if self.rth_safety_factor is not None
            else float(max(getattr(env.cfg, "rth_safety_factor", 1.2), 0.0))
        )
        service_buf = float(max(getattr(env.cfg, "service_battery_buffer", self.service_battery_buffer), 0.0))
        req = float((req + service_buf) * safety)
        batt = float(getattr(a, "battery", 0.0))
        return bool(batt >= req)

    def _finalize_uav_action(
        self,
        env,
        aid: str,
        mode: str,
        candidate: UAVAction,
        goal_id: Optional[str],
        target_agent,
        target_task,
        step_now: int,
        bind_cooldown: int,
        takeoff_cooldown: int,
        bind_reserved: Optional[Dict[str, int]] = None,
    ) -> UAVAction:
        a = env.state.agents[aid]
        cur_xy = self._agent_xy(env, aid)
        out = candidate
        goal_valid = self._goal_valid_for_uav(goal_id, target_agent, target_task)

        # Mode-level coherence guard.
        if mode == self.MODE_DOCKED_STABLE:
            if a.follow_target is not None:
                out = UAVAction(vx=0.0, vy=0.0)
            elif out.bind_truck_id is not None:
                out = UAVAction(vx=0.0, vy=0.0)
        elif mode == self.MODE_SAFE_HOLD:
            if a.follow_target is not None:
                out = UAVAction(vx=0.0, vy=0.0)
            elif out.bind_truck_id is not None or bool(out.takeoff):
                out = UAVAction(vx=0.0, vy=0.0)
        elif mode == self.MODE_READY_FOR_SORTIE:
            if not bool(out.takeoff):
                out = UAVAction(vx=0.0, vy=0.0)
        elif mode == self.MODE_SORTIE_EXECUTION:
            if out.bind_truck_id is not None:
                out = UAVAction(vx=0.0, vy=0.0)

        # Discrete legality gate: bind.
        if out.bind_truck_id is not None:
            bid = str(out.bind_truck_id)
            if a.follow_target is not None and str(a.follow_target) == bid:
                out = UAVAction(vx=0.0, vy=0.0)
            elif mode != self.MODE_RECOVERY_APPROACH:
                out = UAVAction(vx=0.0, vy=0.0)
            elif not self._is_bind_legal(env, aid, bid, cur_xy):
                out = UAVAction(vx=0.0, vy=0.0)
            elif not self._bind_slot_available(env, aid, bid, bind_reserved=bind_reserved):
                out = UAVAction(vx=0.0, vy=0.0)

        # Discrete legality gate: takeoff.
        if bool(out.takeoff):
            if a.follow_target is None:
                out = UAVAction(vx=0.0, vy=0.0)
            elif not self._cooldown_ok(step_now, self._last_takeoff_step.get(str(aid), None), takeoff_cooldown):
                out = UAVAction(vx=0.0, vy=0.0)
            elif goal_id is None:
                out = UAVAction(vx=0.0, vy=0.0)
            elif mode == self.MODE_RECOVERY_APPROACH:
                # Allow takeoff in recovery mode only for truck-to-truck retarget detach.
                transfer_tid = self._planner_transfer_target_truck(env, aid, target_task)
                truck_transfer_ok = bool(
                    transfer_tid is not None
                    and a.follow_target is not None
                    and str(a.follow_target) != str(transfer_tid)
                )
                truck_goal_ok = bool(
                    target_agent is not None
                    and target_agent.kind == AgentKind.TRUCK
                    and a.follow_target is not None
                    and str(a.follow_target) != str(goal_id)
                )
                if not (truck_goal_ok or truck_transfer_ok):
                    out = UAVAction(vx=0.0, vy=0.0)
            elif mode not in (self.MODE_READY_FOR_SORTIE, self.MODE_SORTIE_EXECUTION):
                out = UAVAction(vx=0.0, vy=0.0)
            elif target_task is not None:
                if not self._allow_takeoff_for_task_goal(
                    env, aid, target_task, step_now=step_now
                ):
                    out = UAVAction(vx=0.0, vy=0.0)
            elif target_agent is not None and target_agent.kind == AgentKind.TRUCK:
                # Avoid bind/takeoff oscillation for truck-to-truck retarget.
                out = UAVAction(vx=0.0, vy=0.0)
            else:
                out = UAVAction(vx=0.0, vy=0.0)

        # Continuous legality/safety gate.
        if (out.bind_truck_id is None) and (not bool(out.takeoff)):
            if a.follow_target is not None:
                out = UAVAction(vx=0.0, vy=0.0)
            else:
                v2_delivery_commitment = bool(
                    target_task is not None
                    and self._v2_airborne_delivery_leg_feasible(
                        env, str(aid), target_task
                    )
                )
                goal_missing = goal_id is None or (target_agent is None and target_task is None)
                if goal_missing and (abs(float(out.vx)) > 1e-8 or abs(float(out.vy)) > 1e-8):
                    # If planner has no actionable target, fall back to safe recovery
                    # instead of drifting in-air and depleting battery.
                    out = self._safe_recovery_action(
                        env,
                        aid,
                        step_now=step_now,
                        bind_cooldown=bind_cooldown,
                        bind_reserved=bind_reserved,
                    )
                if target_task is not None:
                    if (
                        not _is_uav_delivery_task(target_task)
                        or target_task.status.name != "PENDING"
                        or self._task_hard_blocked(env, target_task)
                        or (
                            not self._mission_battery_feasible(env, aid, target_task)
                            and not v2_delivery_commitment
                        )
                    ):
                        out = self._safe_recovery_action(
                            env,
                            aid,
                            step_now=step_now,
                            bind_cooldown=bind_cooldown,
                            bind_reserved=bind_reserved,
                        )
                if (
                    not v2_delivery_commitment
                    and not self._return_feasible_after_action(env, aid, out)
                ):
                    out = self._safe_recovery_action(
                        env,
                        aid,
                        step_now=step_now,
                        bind_cooldown=bind_cooldown,
                        bind_reserved=bind_reserved,
                    )
        if (not goal_valid) and (out.bind_truck_id is None) and (not bool(out.takeoff)):
            # Stale-goal guard:
            # keep recovery path valid, but suppress non-recovery stale pursuit.
            if mode != self.MODE_RECOVERY_APPROACH:
                out = UAVAction(vx=0.0, vy=0.0)

        return out

    def _route_plan_v2_truck_action(
        self,
        env,
        aid: str,
        state,
        legal,
        goal_id: Optional[str],
    ) -> Optional[TruckAction]:
        """Execute only the current layer-1 stop."""
        if not er_hlns_route_plan_active(env):
            return None

        stay_reasons = getattr(
            env, "_planner_route_plan_stay_reason_by_agent", None
        )
        if not isinstance(stay_reasons, dict):
            stay_reasons = {}
            env._planner_route_plan_stay_reason_by_agent = stay_reasons

        # Final execution guard for an atomic DIRECT routine commitment.
        # A recovery/launch assist is advisory, while a stocked truck already
        # at its contracted demand node has an immediately executable unload.
        # Resolve that physical service first so a later assist cannot make
        # the truck depart at zero distance and strand the task.
        if state.node is not None:
            for local_task in env.state.tasks.values():
                if (
                    local_task.status != TaskStatus.PENDING
                    or local_task.kind != TaskKind.NORMAL
                    or str(getattr(local_task, "service_mode", "DIRECT")).upper()
                    == "BULK_RELAY"
                    or int(local_task.demand_node) != int(state.node)
                    or str(
                        getattr(local_task, "route_contract_truck", "") or ""
                    )
                    != str(aid)
                    or not bool(
                        env.is_task_serviceable_by_agent(str(aid), local_task)
                    )
                ):
                    continue
                stay_reasons[str(aid)] = (
                    "hold_for_atomic_onsite_normal_unloading"
                )
                return TruckAction(stay=True)

        assist_map = getattr(
            env, "_planner_truck_assist_waypoint_by_truck", {}
        )
        assist = (
            assist_map.get(str(aid), None)
            if isinstance(assist_map, dict)
            else None
        )
        if isinstance(assist, dict) and bool(
            assist.get("route_plan_v2", False)
        ):
            task_id = str(assist.get("task_id", "")).strip()
            task = env.state.tasks.get(task_id, None)
            launch_node = assist.get("launch_node", None)
            safety_recovery_assist = bool(
                str(assist.get("service_mode", "")).strip().upper()
                == "SAFETY_RECOVERY"
            )
            if (
                (
                    safety_recovery_assist
                    or (
                        task is not None
                        and task.status in (TaskStatus.PENDING, TaskStatus.CLAIMED)
                    )
                )
                and launch_node is not None
                and state.node is not None
            ):
                launch_node = int(launch_node)
                if int(state.node) == launch_node:
                    stay_reasons[str(aid)] = (
                        "hold_at_v2_safety_recovery_boundary"
                        if safety_recovery_assist
                        else "hold_at_v2_launch_anchor"
                    )
                    return TruckAction(stay=True)
                current = float(
                    env._decision_shortest_path_distance(
                        int(state.node), launch_node
                    )
                )
                candidates = [
                    (
                        float(
                            env._decision_shortest_path_distance(
                                int(neighbor), launch_node
                            )
                        ),
                        int(neighbor),
                    )
                    for neighbor in legal
                ]
                candidates = [
                    item for item in candidates if np.isfinite(item[0])
                ]
                if np.isfinite(current) and candidates:
                    best_distance, best_neighbor = min(candidates)
                    if best_distance + 1e-9 < current:
                        stay_reasons[str(aid)] = (
                            "moving_to_v2_launch_anchor"
                        )
                        return TruckAction(
                            target_node=int(best_neighbor), stay=False
                        )
                stay_reasons[str(aid)] = "v2_launch_anchor_unreachable"
                return TruckAction(stay=True)

        task = (
            env.state.tasks.get(str(goal_id), None)
            if goal_id is not None
            else None
        )
        if (
            task is not None
            and task.status in (TaskStatus.PENDING, TaskStatus.CLAIMED)
            and task.kind == TaskKind.NORMAL
            and str(getattr(task, "service_mode", "DIRECT")).upper()
            != "BULK_RELAY"
            and state.node is not None
        ):
            if int(state.node) == int(task.demand_node):
                stay_reasons[str(aid)] = "hold_for_normal_unloading"
                return TruckAction(stay=True)
            current = float(
                env._decision_shortest_path_distance(
                    int(state.node), int(task.demand_node)
                )
            )
            candidates = [
                (
                    float(
                        env._decision_shortest_path_distance(
                            int(neighbor), int(task.demand_node)
                        )
                    ),
                    int(neighbor),
                )
                for neighbor in legal
            ]
            candidates = [
                item for item in candidates if np.isfinite(item[0])
            ]
            if np.isfinite(current) and candidates:
                best_distance, best_neighbor = min(candidates)
                if best_distance + 1e-9 < current:
                    stay_reasons[str(aid)] = "moving_to_normal_route_stop"
                    return TruckAction(
                        target_node=int(best_neighbor), stay=False
                    )
            stay_reasons[str(aid)] = "normal_route_stop_unreachable"
            return TruckAction(stay=True)

        # Contract-recovery fallback: a suffix repair may leave an active
        # routine contract owned by this truck without a materialized current
        # RouteStop.  Execute the nearest such owned contract instead of
        # idling; ownership remains exclusive and no task is stolen here.
        enable_v2_orphan_contract_recovery_experiment = False
        if enable_v2_orphan_contract_recovery_experiment and state.node is not None:
            stock = float(
                max(getattr(state, "bulk_inventory_kg_current", 0.0), 0.0)
            )
            owned_normal_candidates = []
            for owned_task in env.state.tasks.values():
                if (
                    owned_task.status not in (TaskStatus.PENDING, TaskStatus.CLAIMED)
                    or owned_task.kind != TaskKind.NORMAL
                    or str(getattr(owned_task, "service_mode", "DIRECT")).upper()
                    == "BULK_RELAY"
                    or str(
                        getattr(owned_task, "route_contract_truck", "") or ""
                    )
                    != str(aid)
                    or stock + 1e-9
                    < float(max(getattr(owned_task, "remaining_demand_kg", 0.0), 0.0))
                ):
                    continue
                road = float(
                    env._decision_shortest_path_distance(
                        int(state.node), int(owned_task.demand_node)
                    )
                )
                if np.isfinite(road):
                    owned_normal_candidates.append(
                        (road, int(owned_task.deadline_step), str(owned_task.task_id), owned_task)
                    )
            if owned_normal_candidates:
                _, _, _, owned_task = min(owned_normal_candidates)
                if int(state.node) == int(owned_task.demand_node):
                    stay_reasons[str(aid)] = "hold_for_owned_normal_contract_recovery"
                    return TruckAction(stay=True)
                current = float(
                    env._decision_shortest_path_distance(
                        int(state.node), int(owned_task.demand_node)
                    )
                )
                candidates = [
                    (
                        float(
                            env._decision_shortest_path_distance(
                                int(neighbor), int(owned_task.demand_node)
                            )
                        ),
                        int(neighbor),
                    )
                    for neighbor in legal
                ]
                candidates = [item for item in candidates if np.isfinite(item[0])]
                if candidates:
                    best_distance, best_neighbor = min(candidates)
                    if best_distance + 1e-9 < current:
                        stay_reasons[str(aid)] = "moving_to_owned_normal_contract_recovery"
                        return TruckAction(target_node=int(best_neighbor), stay=False)
            # Dynamic orphan takeover approach.  Do not steal a live
            # commitment: consider only routine contracts whose published
            # owner has neither the task as its current route head nor as its
            # current goal.  The closest stocked reachable truck alone may
            # approach; the existing onsite-takeover path performs the atomic
            # ownership transfer when it arrives.
            audit = getattr(env, "_planner_route_plan_v2", {})
            routes = audit.get("routes", {}) if isinstance(audit, dict) else {}
            published_goals = dict(
                getattr(env, "_planner_route_plan_goals", {}) or {}
            )
            orphan_candidates = []
            for orphan_task in env.state.tasks.values():
                if (
                    orphan_task.status not in (TaskStatus.PENDING, TaskStatus.CLAIMED)
                    or orphan_task.kind != TaskKind.NORMAL
                    or str(getattr(orphan_task, "service_mode", "DIRECT")).upper()
                    == "BULK_RELAY"
                    or stock + 1e-9
                    < float(max(getattr(orphan_task, "remaining_demand_kg", 0.0), 0.0))
                ):
                    continue
                task_id = str(orphan_task.task_id)
                owner = str(
                    getattr(orphan_task, "route_contract_truck", "") or ""
                )
                owner_route = routes.get(owner, {}) if isinstance(routes, dict) else {}
                owner_active = bool(
                    owner
                    and (
                        str(published_goals.get(owner, "") or "") == task_id
                        or (
                            isinstance(owner_route, dict)
                            and str(owner_route.get("current_task_id", "")) == task_id
                        )
                    )
                )
                if owner_active:
                    continue
                road = float(
                    env._decision_shortest_path_distance(
                        int(state.node), int(orphan_task.demand_node)
                    )
                )
                if not np.isfinite(road):
                    continue
                nearest = (float("inf"), "")
                for candidate_id, candidate_state in env.state.agents.items():
                    if (
                        candidate_state.kind != AgentKind.TRUCK
                        or bool(getattr(candidate_state, "crashed", False))
                        or candidate_state.node is None
                        or float(max(getattr(candidate_state, "bulk_inventory_kg_current", 0.0), 0.0))
                        + 1e-9
                        < float(max(getattr(orphan_task, "remaining_demand_kg", 0.0), 0.0))
                    ):
                        continue
                    candidate_road = float(
                        env._decision_shortest_path_distance(
                            int(candidate_state.node), int(orphan_task.demand_node)
                        )
                    )
                    if (candidate_road, str(candidate_id)) < nearest:
                        nearest = (candidate_road, str(candidate_id))
                if nearest[1] == str(aid):
                    orphan_candidates.append(
                        (road, int(orphan_task.deadline_step), task_id, orphan_task)
                    )
            if orphan_candidates:
                orphan_current, _, _, orphan_task = min(orphan_candidates)
                if int(state.node) == int(orphan_task.demand_node):
                    stay_reasons[str(aid)] = "hold_for_orphan_normal_takeover"
                    return TruckAction(stay=True)
                candidates = [
                    (
                        float(
                            env._decision_shortest_path_distance(
                                int(neighbor), int(orphan_task.demand_node)
                            )
                        ),
                        int(neighbor),
                    )
                    for neighbor in legal
                ]
                candidates = [item for item in candidates if np.isfinite(item[0])]
                if candidates:
                    best_distance, best_neighbor = min(candidates)
                    if best_distance + 1e-9 < float(orphan_current):
                        stay_reasons[str(aid)] = "moving_to_orphan_normal_takeover"
                        return TruckAction(target_node=int(best_neighbor), stay=False)
        stay_reasons[str(aid)] = "no_current_v2_route_stop"
        return TruckAction(stay=True)

    def act(self, env, high_goals: Optional[Dict[str, Optional[str]]] = None):
        actions: Dict[str, object] = {}
        legal_cache = env.legal_actions()
        step_now = int(env.state.step_index)
        self._episode_reset_if_needed(env)
        bind_cooldown = int(max(getattr(env.cfg, "uav_bind_cooldown_steps", self.uav_bind_cooldown_steps), 0))
        takeoff_cooldown = int(
            max(getattr(env.cfg, "uav_takeoff_cooldown_steps", self.uav_takeoff_cooldown_steps), 0)
        )
        bind_reserved: Dict[str, int] = {}

        for aid, a in env.state.agents.items():
            proposed_goal: Optional[str] = None
            authoritative_sortie_task = None
            if (
                a.kind == AgentKind.UAV
                and a.follow_target is None
                and bool(
                    getattr(
                        env.cfg,
                        "uav_authoritative_sortie_goal_precedence_enabled",
                        True,
                    )
                )
            ):
                contract_tid = dict(
                    getattr(env, "_uav_sortie_contract_task", {}) or {}
                ).get(str(aid), None)
                contract_task = (
                    env.state.tasks.get(str(contract_tid), None)
                    if contract_tid is not None
                    else None
                )
                if (
                    contract_task is not None
                    and _is_uav_delivery_task(contract_task)
                    and contract_task.status.name == "PENDING"
                    and bool(env._uav_loaded_for_task(str(aid), contract_task))
                ):
                    try:
                        d_contract = float(
                            env._agent_distance_to_task(str(aid), contract_task)
                        )
                    except Exception:
                        d_contract = float("inf")
                    # The previous implementation applied the authoritative
                    # contract only inside the final one-step capture
                    # envelope.  A loaded UAV could therefore launch safely,
                    # fly most of the outbound leg, and then be retargeted to
                    # another truck/task before reaching that envelope.  The
                    # layer-1 sortie contract is now authoritative for the
                    # complete delivery leg.  Do not re-apply the conservative
                    # pre-launch round-trip feasibility gate after takeoff:
                    # doing so can abandon a valid accepted delivery merely
                    # because its remaining recovery reserve is below the
                    # launch threshold.  Only hard safety states may break the
                    # accepted delivery contract.
                    forced_recovery = bool(
                        dict(
                            getattr(env, "_uav_forced_rth_latch", {}) or {}
                        ).get(str(aid), False)
                    )
                    battery_hard_low = float(getattr(a, "battery", 0.0)) <= float(
                        np.clip(
                            getattr(
                                env.cfg,
                                "uav_low_battery_force_recover_threshold",
                                0.25,
                            ),
                            0.0,
                            1.0,
                        )
                    )
                    if (
                        np.isfinite(d_contract)
                        and not forced_recovery
                        and not battery_hard_low
                    ):
                        authoritative_sortie_task = str(contract_task.task_id)
            if authoritative_sortie_task is not None:
                proposed_goal = str(authoritative_sortie_task)
                incoming = (
                    None
                    if high_goals is None
                    else high_goals.get(str(aid), None)
                )
                if incoming is not None and str(incoming) != proposed_goal:
                    env.uav_authoritative_sortie_goal_override_count = int(
                        getattr(
                            env,
                            "uav_authoritative_sortie_goal_override_count",
                            0,
                        )
                    ) + 1
            elif high_goals is not None and aid in high_goals and high_goals[aid] is not None:
                proposed_goal = str(high_goals[aid])
            else:
                g = env._effective_goals.get(str(aid), env._recommended_goals.get(str(aid), None))
                if g is not None:
                    proposed_goal = str(g)
            goal_id = self._resolve_uav_goal_with_lock(
                env,
                str(aid),
                proposed_goal,
                step_now=int(step_now),
            )

            if a.kind == AgentKind.TRUCK:
                legal = legal_cache[aid]["neighbors"]
                if a.transit is not None:
                    actions[aid] = TruckAction(stay=True)
                    continue
                if int(getattr(a, "truck_replenish_timer", 0)) > 0:
                    actions[aid] = TruckAction(stay=True)
                    continue
                requires_depot = bool(getattr(a, "truck_needs_replenish_flag", False))
                if hasattr(env, "_truck_requires_depot"):
                    requires_depot = bool(env._truck_requires_depot(aid))
                if requires_depot:
                    if int(a.node or 0) == 0:
                        actions[aid] = TruckAction(stay=True)
                        continue
                    if not legal:
                        actions[aid] = TruckAction(stay=True)
                        continue
                    best_nb = min(
                        legal,
                        key=lambda nb: (
                            float(env._decision_shortest_path_distance(int(nb), 0))
                            if hasattr(env, "_decision_shortest_path_distance")
                            else float(env.topology.shortest_path_distance(int(nb), 0, ignore_blocked=False))
                        ),
                    )
                    actions[aid] = TruckAction(target_node=int(best_nb), stay=False)
                    continue
                if not legal:
                    actions[aid] = TruckAction(stay=True)
                    continue

                route_v2_action = self._route_plan_v2_truck_action(
                    env,
                    str(aid),
                    a,
                    legal,
                    goal_id,
                )
                if route_v2_action is not None:
                    actions[aid] = route_v2_action
                    continue

                # Dual rendezvous recovery support: keep it strictly conditional so
                # truck task throughput is not starved by mild recovery requests.
                support_nb = None
                hard_recovery_active = bool(getattr(env, "_has_hard_recovery_uav", lambda: False)())
                assigned_airborne_hard_recovery = bool(
                    getattr(env, "_truck_has_assigned_airborne_hard_recovery_request", lambda *_args, **_kwargs: False)(
                        str(aid)
                    )
                )
                has_serviceable_normal = False
                for cand in env.state.tasks.values():
                    if cand.status != TaskStatus.PENDING or cand.kind != TaskKind.NORMAL:
                        continue
                    if hasattr(env, "is_task_serviceable_by_agent") and (not bool(env.is_task_serviceable_by_agent(str(aid), cand))):
                        continue
                    d_cand = float(
                        env._decision_shortest_path_distance(int(a.node), int(cand.demand_node))
                        if hasattr(env, "_decision_shortest_path_distance")
                        else env.topology.shortest_path_distance(int(a.node), int(cand.demand_node), ignore_blocked=False)
                    )
                    if np.isfinite(d_cand):
                        has_serviceable_normal = True
                        break
                if bool(getattr(env.cfg, "truck_support_uav_recovery_enabled", True)) and hasattr(env, "_truck_recovery_support_target"):
                    support_nb = env._truck_recovery_support_target(str(aid), [int(x) for x in legal])
                routine_goal_protected = bool(
                    getattr(env, "_truck_routine_goal_support_protected", lambda *_args, **_kwargs: False)(
                        str(aid),
                        [int(x) for x in legal],
                    )
                )
                tc_override_allowed = False
                if routine_goal_protected and support_nb is not None:
                    multiround_commitment = bool(
                        getattr(
                            env,
                            "_truck_active_multiround_routine_commitment",
                            lambda *_args, **_kwargs: None,
                        )(str(aid))
                        is not None
                    )
                    if not multiround_commitment:
                        tc_override_allowed = bool(
                            getattr(env, "_routine_protection_tc_override_allowed", lambda *_args, **_kwargs: (False, {}))(
                                str(aid),
                                int(support_nb),
                            )[0]
                        )
                if (
                    support_nb is not None
                    and (hard_recovery_active or (not has_serviceable_normal))
                    and ((not routine_goal_protected) or assigned_airborne_hard_recovery or tc_override_allowed)
                ):
                    actions[aid] = TruckAction(target_node=int(support_nb), stay=False)
                    continue

                assist_map = getattr(env, "_planner_truck_assist_waypoint_by_truck", {})
                assist = assist_map.get(str(aid), None) if isinstance(assist_map, dict) else None
                if isinstance(assist, dict):
                    assist_task_id = str(assist.get("task_id", "")).strip()
                    normal_goal_task_id = str(assist.get("normal_goal_task_id", "")).strip()
                    idle_support = bool(assist.get("idle_support", False))
                    launch_node_raw = assist.get("launch_node", None)
                    assist_task = env.state.tasks.get(assist_task_id, None) if assist_task_id else None
                    goal_matches_assist = bool(
                        idle_support
                        or (
                            goal_id is not None
                            and normal_goal_task_id
                            and str(goal_id) == normal_goal_task_id
                        )
                    )
                    if (
                        assist_task is not None
                        and assist_task.status == TaskStatus.PENDING
                        and assist_task.kind == TaskKind.EMERGENCY
                        and goal_matches_assist
                        and launch_node_raw is not None
                        and a.node is not None
                    ):
                        launch_node = int(launch_node_raw)
                        try:
                            d_current = float(env._decision_shortest_path_distance(int(a.node), int(launch_node)))
                            near_radius = float(max(getattr(env.cfg, "uav_delivery_radius_m", 40.0), 1.0))
                            if np.isfinite(d_current) and d_current > near_radius:
                                best_nb = min(
                                    legal,
                                    key=lambda nb: float(env._decision_shortest_path_distance(int(nb), int(launch_node))),
                                )
                                best_d = float(env._decision_shortest_path_distance(int(best_nb), int(launch_node)))
                                if np.isfinite(best_d) and best_d + 1e-9 < d_current:
                                    cur_count = int(getattr(env, "truck_uav_assist_waypoint_move_count_total", 0))
                                    env.truck_uav_assist_waypoint_move_count_total = int(cur_count + 1)
                                    actions[aid] = TruckAction(target_node=int(best_nb), stay=False)
                                    continue
                        except Exception:
                            pass

                # Hard idle-support rule: a truck without its own delivery
                # goal must not wait while a docked/assigned UAV still has a
                # pending emergency delivery.  Move the truck to the nearest
                # road-reachable safe launch area; this is intentionally
                # independent of the soft attraction score.
                if goal_id is None and a.node is not None:
                    support_choices = []
                    for uid, us in env.state.agents.items():
                        if us.kind != AgentKind.UAV or bool(getattr(us, "crashed", False)):
                            continue
                        gid_u = None
                        if high_goals is not None:
                            gid_u = high_goals.get(str(uid), None)
                        if gid_u is None:
                            gid_u = env._effective_goals.get(str(uid), env._recommended_goals.get(str(uid), None))
                        task_u = env.state.tasks.get(str(gid_u), None) if gid_u is not None else None
                        if task_u is None or task_u.status != TaskStatus.PENDING or task_u.kind != TaskKind.EMERGENCY:
                            continue
                        # A support move is meaningful only for a UAV that is
                        # physically docked on this truck and can launch from
                        # it.  Do not send unrelated empty trucks wandering
                        # toward a task without a concrete UAV contract.
                        own_uav = str(getattr(us, "follow_target", "") or "") == str(aid)
                        if not own_uav:
                            continue
                        task_xy = env._node_xy(int(task_u.demand_node))
                        max_sortie = float(max(getattr(env.cfg, "uav_max_sortie_m", 6000.0), 1.0))
                        recovery_buf = float(max(getattr(env.cfg, "uav_recovery_distance_buffer_m", 600.0), 0.0))
                        launch_radius = float(max((0.92 * max_sortie - recovery_buf) / 2.0, 1.0))
                        best_stage, best_d, best_key = None, float("inf"), (float("inf"), float("inf"))
                        for nid in env.topology.nodes:
                            nxy = env._node_xy(int(nid))
                            air_d = float(np.hypot(float(nxy[0]) - float(task_xy[0]), float(nxy[1]) - float(task_xy[1])))
                            if air_d > launch_radius:
                                continue
                            d = float(env._decision_shortest_path_distance(int(a.node), int(nid)))
                            cand_key = (air_d, d)
                            if np.isfinite(d) and cand_key < best_key:
                                best_stage, best_d, best_key = int(nid), d, cand_key
                        if best_stage is not None:
                            support_choices.append((0 if own_uav else 1, best_d, str(uid), str(task_u.task_id), int(best_stage)))
                    if support_choices:
                        _, _, _, _, stage_node = min(support_choices)
                        try:
                            d_current = float(env._decision_shortest_path_distance(int(a.node), int(stage_node)))
                            best_nb = min(legal, key=lambda nb: float(env._decision_shortest_path_distance(int(nb), int(stage_node))))
                            best_d = float(env._decision_shortest_path_distance(int(best_nb), int(stage_node)))
                            if np.isfinite(best_d) and best_d + 1e-9 < d_current:
                                cur_count = int(getattr(env, "truck_uav_assist_waypoint_move_count_total", 0))
                                env.truck_uav_assist_waypoint_move_count_total = int(cur_count + 1)
                                actions[aid] = TruckAction(target_node=int(best_nb), stay=False)
                                continue
                        except Exception:
                            pass

                t = env.state.tasks.get(goal_id) if goal_id is not None else None
                if t is not None and hasattr(env, "_decision_shortest_path_distance") and a.node is not None:
                    d_cur = float(env._decision_shortest_path_distance(int(a.node), int(t.demand_node)))
                    if not np.isfinite(d_cur):
                        t = None
                if t is None:
                    # Anti-idle fallback: when high-level leaves truck without explicit task,
                    # greedily pick nearest reachable serviceable pending task first.
                    nearest_task = None
                    nearest_dist = float("inf")
                    best_priority = 999
                    has_pending_normal = any(
                        t.status == TaskStatus.PENDING and t.kind == TaskKind.NORMAL
                        for t in env.state.tasks.values()
                    )
                    for cand in env.state.tasks.values():
                        if cand.status != TaskStatus.PENDING:
                            continue
                        if hasattr(env, "is_task_serviceable_by_agent") and (not bool(env.is_task_serviceable_by_agent(str(aid), cand))):
                            continue
                        d_cand = float(
                            env._decision_shortest_path_distance(int(a.node), int(cand.demand_node))
                            if hasattr(env, "_decision_shortest_path_distance")
                            else env.topology.shortest_path_distance(int(a.node), int(cand.demand_node), ignore_blocked=False)
                        )
                        if not np.isfinite(d_cand):
                            continue
                        if cand.kind == TaskKind.NORMAL:
                            priority = 0
                        elif has_pending_normal:
                            priority = 2
                        else:
                            priority = 1
                        better = False
                        if int(priority) < int(best_priority):
                            better = True
                        elif int(priority) == int(best_priority) and d_cand + 1e-9 < nearest_dist:
                            better = True
                        if better:
                            best_priority = int(priority)
                            nearest_dist = float(d_cand)
                            nearest_task = cand

                    if nearest_task is not None:
                        t = nearest_task
                    else:
                        fallback_nb = None
                        island_ids = set(getattr(env, "_current_island_emergency_task_ids", lambda: set())())
                        if island_ids and hasattr(env, "_truck_island_forward_support_target"):
                            fallback_nb = env._truck_island_forward_support_target(str(aid), [int(x) for x in legal], island_ids)
                        if fallback_nb is None and hasattr(env, "_truck_shared_map_relevant_frontier_target"):
                            fallback_nb = env._truck_shared_map_relevant_frontier_target(str(aid), [int(x) for x in legal])
                        if fallback_nb is None:
                            actions[aid] = TruckAction(stay=True)
                        else:
                            actions[aid] = TruckAction(target_node=int(fallback_nb), stay=False)
                        continue
                best_nb = min(
                    legal,
                    key=lambda nb: (
                        float(env._decision_shortest_path_distance(int(nb), int(t.demand_node)))
                        if hasattr(env, "_decision_shortest_path_distance")
                        else float(env.topology.shortest_path_distance(int(nb), int(t.demand_node), ignore_blocked=False))
                    ),
                )
                actions[aid] = TruckAction(target_node=int(best_nb), stay=False)
                continue

            # UAV execution with final legality/safety gate.
            cur_follow_obs = None if a.follow_target is None else str(a.follow_target)
            prev_follow_obs = self._last_follow_target_observed.get(str(aid), None)
            if (
                prev_follow_obs is not None
                and cur_follow_obs is None
                and (not bool(self._takeoff_cmd_latch.get(str(aid), False)))
            ):
                # Count unbind transitions as executed takeoff events when they
                # were not triggered by our explicit takeoff command.
                self.takeoff_count_episode += 1
            self._takeoff_cmd_latch[str(aid)] = False

            if a.crashed:
                self._set_uav_mode(str(aid), self.MODE_SAFE_HOLD, step_now=step_now)
                self._docked_steps[str(aid)] = 0
                actions[aid] = UAVAction(vx=0.0, vy=0.0)
                self._last_follow_target_observed[str(aid)] = cur_follow_obs
                continue

            target_agent = env.state.agents.get(goal_id) if goal_id is not None else None
            target_task = env.state.tasks.get(goal_id) if goal_id is not None else None
            cur_xy = self._agent_xy(env, aid)
            vmax = float(env.cfg.uav_max_speed_mps)
            can_takeoff = self._cooldown_ok(
                step_now,
                self._last_takeoff_step.get(str(aid), None),
                takeoff_cooldown,
            )
            if a.follow_target is not None:
                self._docked_steps[str(aid)] = int(self._docked_steps.get(str(aid), 0) + 1)
            else:
                self._docked_steps[str(aid)] = 0

            mode = self._decide_uav_mode(
                env,
                str(aid),
                goal_id=goal_id,
                target_agent=target_agent,
                target_task=target_task,
                step_now=step_now,
            )
            self._set_uav_mode(str(aid), mode, step_now=step_now)

            if mode == self.MODE_DOCKED_STABLE:
                self.docked_stable_steps_episode += 1
            if mode == self.MODE_SAFE_HOLD:
                self.safe_hold_steps_episode += 1

            cand = UAVAction(vx=0.0, vy=0.0)
            recovery_truck_id = self._resolve_recovery_truck(env, str(aid), goal_id, target_agent)

            if mode == self.MODE_RECOVERY_APPROACH:
                transfer_tid = self._planner_transfer_target_truck(env, str(aid), target_task)
                if (
                    a.follow_target is not None
                    and transfer_tid is not None
                    and str(a.follow_target) != str(transfer_tid)
                ):
                    cand = UAVAction(takeoff=True)
                elif a.follow_target is not None and recovery_truck_id is not None and str(a.follow_target) != str(recovery_truck_id):
                    # Keep ordinary recovery conservative; only explicit planner
                    # transfer hints may trigger truck-to-truck detach.
                    cand = UAVAction(vx=0.0, vy=0.0)
                else:
                    cand = self._safe_recovery_action(
                        env,
                        str(aid),
                        step_now=step_now,
                        bind_cooldown=bind_cooldown,
                        bind_reserved=bind_reserved,
                        preferred_truck_id=recovery_truck_id,
                    )
            elif mode == self.MODE_READY_FOR_SORTIE:
                if a.follow_target is not None and can_takeoff and target_task is not None:
                    if self._allow_takeoff_for_task_goal(env, str(aid), target_task, step_now=step_now):
                        cand = UAVAction(takeoff=True)
                    else:
                        cand = UAVAction(vx=0.0, vy=0.0)
                else:
                    cand = UAVAction(vx=0.0, vy=0.0)
            elif mode == self.MODE_SORTIE_EXECUTION:
                if target_task is None or not _is_uav_delivery_task(target_task):
                    cand = UAVAction(vx=0.0, vy=0.0)
                elif a.follow_target is not None:
                    if can_takeoff and self._allow_takeoff_for_task_goal(env, str(aid), target_task, step_now=step_now):
                        cand = UAVAction(takeoff=True)
                    else:
                        cand = UAVAction(vx=0.0, vy=0.0)
                else:
                    task_xy = env._node_xy(int(target_task.demand_node))
                    vx, vy = self._full_speed_to(cur_xy, task_xy, vmax=vmax)
                    cand = UAVAction(vx=vx, vy=vy)
            elif mode == self.MODE_SAFE_HOLD:
                if a.follow_target is not None:
                    cand = UAVAction(vx=0.0, vy=0.0)
                else:
                    cand = UAVAction(vx=0.0, vy=0.0)
            else:
                # DOCKED_STABLE default.
                cand = UAVAction(vx=0.0, vy=0.0)

            final_act = self._finalize_uav_action(
                env=env,
                aid=str(aid),
                mode=mode,
                candidate=cand,
                goal_id=goal_id,
                target_agent=target_agent,
                target_task=target_task,
                step_now=step_now,
                bind_cooldown=bind_cooldown,
                takeoff_cooldown=takeoff_cooldown,
                bind_reserved=bind_reserved,
            )

            if final_act.bind_truck_id is not None:
                self._last_bind_step[str(aid)] = int(step_now)
                self.bind_count_episode += 1
                bt = str(final_act.bind_truck_id)
                bind_reserved[bt] = int(bind_reserved.get(bt, 0) + 1)
            if bool(final_act.takeoff):
                self._last_takeoff_step[str(aid)] = int(step_now)
                self.takeoff_count_episode += 1
                self._takeoff_cmd_latch[str(aid)] = True

            actions[aid] = final_act
            self._last_follow_target_observed[str(aid)] = cur_follow_obs

        pre_dispatch = getattr(env, "pre_dispatch_validate_actions", None)
        if callable(pre_dispatch):
            return pre_dispatch(actions)
        return actions


class AttentionGuidedLowLevelPolicy(RuleBasedLowLevelPolicy):
    """
    Rule-based execution with attention-driven goal selection in the forward chain.
    """

    def __init__(
        self,
        obs_dim: int = 20,
        hidden_dim: int = 64,
        seed: int = 0,
        use_hetgat: bool = True,
        enable_rth_mask: bool = True,
    ):
        super().__init__(seed=seed)
        torch.manual_seed(seed)
        self.use_hetgat = bool(use_hetgat)
        self.enable_rth_mask = bool(enable_rth_mask)
        self.attn = TaskAttentionModule(
            agent_dim=obs_dim,
            task_dim=5,
            hidden_dim=hidden_dim,
        )
        self.last_attention_summary: Dict[str, float] = {}

    def infer_attention_goals(
        self,
        env,
        fallback_goals: Optional[Dict[str, Optional[str]]] = None,
    ) -> Tuple[Dict[str, Optional[str]], Dict[str, float]]:
        obs = env.observe()
        task_mat = env.observe_task_matrix()
        task_slots = env.observe_task_slots() if hasattr(env, "observe_task_slots") else {}
        goals: Dict[str, Optional[str]] = {}
        entropies: List[float] = []
        rank_map: Dict[str, List[Tuple[float, Optional[str]]]] = {}
        forced_rth_mask_count = 0
        use_hetgat = bool(getattr(env.cfg, "use_hetgat", self.use_hetgat))
        enable_rth_mask = bool(
            getattr(env.cfg, "enable_rth_mask", self.enable_rth_mask)
        )

        def _agent_xy(aid: str) -> Tuple[float, float]:
            st = env.state.agents[aid]
            if st.pos_xy is not None:
                return float(st.pos_xy[0]), float(st.pos_xy[1])
            if hasattr(env, "_node_xy"):
                return env._node_xy(int(st.node or 0))
            return 0.0, 0.0

        def _nearest_truck_for(aid: str) -> Tuple[Optional[str], float]:
            ax, ay = _agent_xy(aid)
            best_id: Optional[str] = None
            best_d = float("inf")
            for tid, ts in env.state.agents.items():
                if ts.kind != AgentKind.TRUCK:
                    continue
                tx, ty = _agent_xy(str(tid))
                d = float(np.hypot(ax - tx, ay - ty))
                if d < best_d:
                    best_d = d
                    best_id = str(tid)
            return best_id, best_d

        def _required_rth_battery(agent_state, dist_to_truck: float) -> float:
            # Conservative per-meter energy estimate from configured physics coefficients.
            base_discharge_per_m = float(max(getattr(env.cfg, "uav_flight_discharge_per_m", 1e-6), 1e-6))
            headwind_coeff = float(max(getattr(env.cfg, "uav_headwind_energy_coeff", 0.04), 0.0))
            rain_coeff = float(max(getattr(env.cfg, "uav_rain_energy_coeff", 0.02), 0.0))
            base_wind = float(max(getattr(env.cfg, "base_wind_mps", 0.0), 0.0))
            base_rain = float(max(getattr(env.cfg, "base_rainfall_mmh", 0.0), 0.0))
            cargo_unit_kg = float(max(getattr(env.cfg, "cargo_unit_kg", 200.0), 1e-6))
            m_load_kg = float(max(getattr(agent_state, "cargo", 0.0), 0.0)) * cargo_unit_kg
            load_factor = 1.0 + 0.018 * m_load_kg
            weather_factor = 1.0 + headwind_coeff * base_wind + rain_coeff * base_rain
            safe_discharge_rate = base_discharge_per_m * weather_factor * load_factor
            return float(max(0.0, dist_to_truck) * safe_discharge_rate)

        for aid in env.state.agents:
            obs_vec = torch.tensor(obs[aid], dtype=torch.float32).view(1, 1, -1)
            task_tensor = torch.tensor(task_mat[aid], dtype=torch.float32).view(1, -1, 5)
            slot_ids = task_slots.get(aid, [None] * len(task_mat[aid]))
            mask_vals = [1.0 if sid is not None else 0.0 for sid in slot_ids]
            agent_state = env.state.agents[aid]
            if agent_state.kind == AgentKind.UAV and bool(agent_state.crashed):
                # Crashed UAV should not produce actionable goals.
                goals[aid] = None
                rank_map[aid] = []
                continue

            # Dynamic attention masking for alive UAV:
            # 1) cargo-empty interception
            # 2) return-to-truck feasibility interception based on energy budget
            if agent_state.kind == AgentKind.UAV and not bool(agent_state.crashed):
                nearest_truck_id, dist_to_truck = _nearest_truck_for(aid)
                force_rth = bool(float(agent_state.cargo) <= 0.0)
                if (not force_rth) and nearest_truck_id is not None and np.isfinite(dist_to_truck):
                    required_battery = _required_rth_battery(agent_state, float(dist_to_truck))
                    safety_factor = float(max(getattr(env.cfg, "rth_safety_factor", 1.2), 0.0))
                    if float(agent_state.battery) < float(required_battery * safety_factor):
                        force_rth = True
                if enable_rth_mask and force_rth:
                    forced_rth_mask_count += 1
                    for i, sid in enumerate(slot_ids):
                        sid_str = "" if sid is None else str(sid)
                        if sid is None:
                            mask_vals[i] = 0.0
                            continue
                        if sid_str.startswith("truck"):
                            # Keep only nearest truck slot visible.
                            mask_vals[i] = 1.0 if (nearest_truck_id is not None and sid_str == nearest_truck_id) else 0.0
                        else:
                            mask_vals[i] = 0.0
                    # Safety fallback: if nearest truck slot not present in slots, keep any truck slot.
                    if float(sum(mask_vals)) <= 0.0:
                        for i, sid in enumerate(slot_ids):
                            sid_str = "" if sid is None else str(sid)
                            if sid is not None and sid_str.startswith("truck"):
                                mask_vals[i] = 1.0
            task_mask = torch.tensor(mask_vals, dtype=torch.float32).view(1, -1)

            if use_hetgat:
                _, weights = self.attn(obs_vec, task_tensor, task_mask=task_mask)
                w = weights[0, 0]  # [T]
            else:
                # Pooled-features fallback: no graph attention,
                # simple feature pooling/linear scoring on task slots.
                t = task_tensor[0]  # [T,5]
                dist_norm = t[:, 2]
                emer = t[:, 3]
                is_rec = t[:, 4]
                score = (-1.2 * dist_norm) + (0.45 * emer) + (0.15 * is_rec)
                score = score + (task_mask[0] - 1.0) * 1e9
                w = torch.softmax(score, dim=0)
            top_idx = int(torch.argmax(w).item())
            prob = torch.clamp(w, min=1e-9)
            entropy = float((-prob * torch.log(prob)).sum().item())
            entropies.append(entropy)
            ranked: List[Tuple[float, Optional[str]]] = []
            for i, sid in enumerate(slot_ids):
                if sid is None:
                    continue
                tid = str(sid)
                t = env.state.tasks.get(tid)
                if t is not None:
                    if (
                        env.state.agents[aid].kind == AgentKind.UAV
                        and t.kind != TaskKind.EMERGENCY
                    ):
                        continue
                    ranked.append((float(w[i].item()), tid))
                    continue
                # Virtual slot: truck agent id (e.g., "truck_0")
                ag = env.state.agents.get(tid)
                if ag is not None and ag.kind == AgentKind.TRUCK:
                    ranked.append((float(w[i].item()), tid))
            ranked.sort(key=lambda x: x[0], reverse=True)
            # Keep top_idx computation for diagnostics consistency.
            _ = top_idx
            rank_map[aid] = ranked

        used: set = set()
        ordered_agents = sorted(
            env.state.agents.keys(),
            key=lambda a: 0 if env.state.agents[a].kind == AgentKind.UAV else 1,
        )
        for aid in ordered_agents:
            st = env.state.agents[aid]
            if st.kind == AgentKind.UAV and bool(st.crashed):
                goals[aid] = None
                continue
            picked: Optional[str] = None
            for _, tid in rank_map.get(aid, []):
                if tid in env.state.tasks and tid in used:
                    continue
                picked = tid
                break
            if picked is None and fallback_goals is not None:
                fb = fallback_goals.get(aid, None)
                if fb is not None:
                    fb = str(fb)
                    t = env.state.tasks.get(fb)
                    if t is not None:
                        if env.state.agents[aid].kind == AgentKind.UAV and t.kind != TaskKind.EMERGENCY:
                            fb = None
                    if fb is not None and ((fb not in env.state.tasks) or (fb not in used)):
                        picked = fb
            goals[aid] = picked
            if picked is not None and picked in env.state.tasks:
                used.add(picked)

        summary = {
            "attention_entropy_mean": float(np.mean(entropies)) if entropies else 0.0,
            "attention_agents": float(len(entropies)),
            "forced_rth_mask_count": float(forced_rth_mask_count),
        }
        self.last_attention_summary = summary
        return goals, summary


def build_task_feature_matrix(env) -> Dict[str, List[float]]:
    """
    Per-agent flattened task features [dx, dy, dist, emer_flag, is_rec, ...].
    """
    out: Dict[str, List[float]] = {}
    tasks = [
        t
        for t in env.state.tasks.values()
        if t.status.name in ("PENDING", "CLAIMED")
    ]
    for aid in env.state.agents:
        feats: List[float] = []
        rec_tid = env._effective_goals.get(str(aid), env._recommended_goals.get(str(aid), None))
        for t in tasks:
            dx, dy, d = env._agent_task_rel(aid, t)
            feats.extend(
                [
                    float(dx),
                    float(dy),
                    float(np.clip(d, 0.0, 1.0)),
                    1.0 if t.kind.name == "EMERGENCY" else 0.0,
                    1.0 if (rec_tid is not None and str(t.task_id) == str(rec_tid)) else 0.0,
                ]
            )
        out[aid] = feats
    return out


class _LowLevelGaussianActorCritic(nn.Module):
    """Continuous Gaussian actor-critic for UAV low-level control."""

    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, 2)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((2,), -0.7))

    def forward(self, obs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        h = self.encoder(obs)
        mu = self.mu_head(h)
        value = self.value_head(h).squeeze(-1)
        log_std = self.log_std.unsqueeze(0).expand_as(mu)
        return mu, log_std, value


class LearnableLowLevelPolicy:
    """
    Learnable UAV low-level policy.
    - UAV: Gaussian continuous control [vx, vy] with tanh squashing.
    - Truck: keep topological discrete heuristic execution.
    """

    def __init__(
        self,
        seed: int = 0,
        obs_dim: int = 12,
        hidden_dim: int = 128,
        device: str = "cpu",
    ):
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.obs_dim = int(obs_dim)
        self.model = _LowLevelGaussianActorCritic(obs_dim=self.obs_dim, hidden_dim=hidden_dim).to(
            self.device
        )
        self._truck_policy = RuleBasedLowLevelPolicy(seed=seed)
        self._eps = 1e-6
        # Per-UAV watchdog state for truck-goal soft guidance.
        self._truck_away_streak: Dict[str, int] = {}

    def parameters(self):
        return self.model.parameters()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def _agent_xy(self, env, aid: str) -> Tuple[float, float]:
        st = env.state.agents[aid]
        if st.pos_xy is not None:
            return float(st.pos_xy[0]), float(st.pos_xy[1])
        return env._node_xy(int(st.node or 0))

    @staticmethod
    def _full_speed_to(
        src: Tuple[float, float],
        dst: Tuple[float, float],
        vmax: float,
    ) -> Tuple[float, float]:
        dx = float(dst[0] - src[0])
        dy = float(dst[1] - src[1])
        norm = float(np.hypot(dx, dy))
        if norm <= 1e-6:
            return 0.0, 0.0
        return float(dx / norm * vmax), float(dy / norm * vmax)

    @staticmethod
    def _command_to_target(
        src: Tuple[float, float],
        dst: Tuple[float, float],
        vmax: float,
        dt_seconds: float,
    ) -> Tuple[float, float]:
        """
        Produce a velocity command that reaches the target exactly in one step
        when physically possible, otherwise saturates at vmax.
        """
        dx = float(dst[0] - src[0])
        dy = float(dst[1] - src[1])
        dist = float(np.hypot(dx, dy))
        if dist <= 1e-6:
            return 0.0, 0.0
        req_speed = float(dist / max(float(dt_seconds), 1e-6))
        cmd_speed = float(min(max(req_speed, 0.0), max(float(vmax), 1e-6)))
        return float(dx / dist * cmd_speed), float(dy / dist * cmd_speed)

    @staticmethod
    def _blend_to_goal_if_wrong_way(
        cmd_xy: Tuple[float, float],
        src_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        vmax: float,
        min_cosine: float,
        alpha: float,
    ) -> Tuple[float, float]:
        """
        Soft anti-wrong-way guard:
        if policy command points away from goal, blend a portion of goal-directed
        command. This keeps policy autonomy while avoiding pathological drift.
        """
        vx_cmd, vy_cmd = float(cmd_xy[0]), float(cmd_xy[1])
        gx, gy = float(goal_xy[0] - src_xy[0]), float(goal_xy[1] - src_xy[1])
        gnorm = float(np.hypot(gx, gy))
        vnorm = float(np.hypot(vx_cmd, vy_cmd))
        if gnorm <= 1e-6 or vnorm <= 1e-6:
            return vx_cmd, vy_cmd
        cos = float((vx_cmd * gx + vy_cmd * gy) / max(vnorm * gnorm, 1e-6))
        if cos >= float(min_cosine):
            return vx_cmd, vy_cmd
        ax = float(np.clip(alpha, 0.0, 1.0))
        gdx, gdy = LearnableLowLevelPolicy._full_speed_to(src_xy, goal_xy, vmax=max(vmax, 1e-6))
        bx = (1.0 - ax) * vx_cmd + ax * gdx
        by = (1.0 - ax) * vy_cmd + ax * gdy
        bnorm = float(np.hypot(bx, by))
        if bnorm > float(vmax) and bnorm > 1e-6:
            scale = float(vmax / bnorm)
            bx *= scale
            by *= scale
        return float(bx), float(by)

    @staticmethod
    def _alignment_cos(
        cmd_xy: Tuple[float, float],
        src_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
    ) -> float:
        vx_cmd, vy_cmd = float(cmd_xy[0]), float(cmd_xy[1])
        gx, gy = float(goal_xy[0] - src_xy[0]), float(goal_xy[1] - src_xy[1])
        gnorm = float(np.hypot(gx, gy))
        vnorm = float(np.hypot(vx_cmd, vy_cmd))
        if gnorm <= 1e-6 or vnorm <= 1e-6:
            return 1.0
        return float((vx_cmd * gx + vy_cmd * gy) / max(vnorm * gnorm, 1e-6))

    @staticmethod
    def _enforce_min_cruise_speed(
        cmd_xy: Tuple[float, float],
        src_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        vmax: float,
        min_speed: float,
    ) -> Tuple[float, float]:
        # Enforce a minimum cruise speed in far field to reduce dithering/hover drift.
        v_cap = float(max(vmax, 1e-6))
        v_floor = float(np.clip(min_speed, 0.0, v_cap))
        if v_floor <= 1e-9:
            return float(cmd_xy[0]), float(cmd_xy[1])

        vx_cmd, vy_cmd = float(cmd_xy[0]), float(cmd_xy[1])
        speed = float(np.hypot(vx_cmd, vy_cmd))
        if speed >= v_floor - 1e-9:
            return vx_cmd, vy_cmd

        gx = float(goal_xy[0] - src_xy[0])
        gy = float(goal_xy[1] - src_xy[1])
        gnorm = float(np.hypot(gx, gy))
        if gnorm <= 1e-6:
            return vx_cmd, vy_cmd

        if speed <= 1e-9:
            ux = gx / gnorm
            uy = gy / gnorm
            return float(ux * v_floor), float(uy * v_floor)

        ux = vx_cmd / speed
        uy = vy_cmd / speed
        return float(ux * v_floor), float(uy * v_floor)

    def _required_rth_battery(
        self,
        env,
        aid: str,
        dist_to_truck: float,
    ) -> float:
        s = env.state.agents[aid]
        base_discharge_per_m = float(
            max(getattr(env.cfg, "uav_flight_discharge_per_m", 1e-6), 1e-6)
        )
        headwind_coeff = float(
            max(getattr(env.cfg, "uav_headwind_energy_coeff", 0.04), 0.0)
        )
        rain_coeff = float(max(getattr(env.cfg, "uav_rain_energy_coeff", 0.02), 0.0))
        base_wind = float(max(getattr(env.cfg, "base_wind_mps", 0.0), 0.0))
        base_rain = float(max(getattr(env.cfg, "base_rainfall_mmh", 0.0), 0.0))
        cargo_unit_kg = float(max(getattr(env.cfg, "cargo_unit_kg", 200.0), 1e-6))
        m_load_kg = float(max(getattr(s, "cargo", 0.0), 0.0)) * cargo_unit_kg
        load_factor = 1.0 + 0.018 * m_load_kg
        weather_factor = 1.0 + headwind_coeff * base_wind + rain_coeff * base_rain
        safe_discharge_rate = base_discharge_per_m * weather_factor * load_factor
        return float(max(0.0, dist_to_truck) * safe_discharge_rate)

    def _resolve_goal_xy(
        self, env, aid: str, goal_id: Optional[str]
    ) -> Tuple[float, float, float, float, float]:
        """
        Returns:
            (goal_x, goal_y, goal_is_truck, goal_is_emergency, goal_valid)
        """
        if goal_id is None:
            ax, ay = self._agent_xy(env, aid)
            return float(ax), float(ay), 0.0, 0.0, 0.0
        tid = str(goal_id)
        ag = env.state.agents.get(tid)
        if ag is not None and ag.kind == AgentKind.TRUCK:
            tx, ty = self._agent_xy(env, tid)
            return float(tx), float(ty), 1.0, 0.0, 1.0
        t = env.state.tasks.get(tid)
        if t is not None:
            n = env.topology.nodes[int(t.demand_node)]
            emer = 1.0 if t.kind == TaskKind.EMERGENCY else 0.0
            return float(n.x), float(n.y), 0.0, float(emer), 1.0
        ax, ay = self._agent_xy(env, aid)
        return float(ax), float(ay), 0.0, 0.0, 0.0

    def _build_uav_obs(self, env, aid: str, goal_id: Optional[str]) -> np.ndarray:
        s = env.state.agents[aid]
        ax, ay = self._agent_xy(env, aid)
        gx, gy, goal_is_truck, goal_is_emergency, goal_valid = self._resolve_goal_xy(
            env, aid, goal_id
        )
        dx = float(gx - ax)
        dy = float(gy - ay)
        dist = float(np.hypot(dx, dy))
        vmax = float(max(getattr(env.cfg, "uav_max_speed_mps", 1.0), 1e-6))
        wx, wy = env.hazards.wind_vector_at((float(ax), float(ay)))
        vx = float(s.vel_xy[0]) if s.vel_xy is not None else 0.0
        vy = float(s.vel_xy[1]) if s.vel_xy is not None else 0.0
        cargo_cap = float(max(getattr(env.cfg, "uav_cargo_capacity_units", 1.0), 1e-6))
        obs = np.array(
            [
                float(np.clip(dx / 3000.0, -1.0, 1.0)),
                float(np.clip(dy / 3000.0, -1.0, 1.0)),
                float(np.clip(dist / 3000.0, 0.0, 1.0)),
                float(np.clip(wx / 20.0, -1.0, 1.0)),
                float(np.clip(wy / 20.0, -1.0, 1.0)),
                float(np.clip(vx / vmax, -1.0, 1.0)),
                float(np.clip(vy / vmax, -1.0, 1.0)),
                float(np.clip(s.battery, 0.0, 1.0)),
                float(np.clip(float(s.cargo) / cargo_cap, 0.0, 1.0)),
                float(1.0 if s.follow_target is not None else 0.0),
                float(goal_is_truck),
                float(goal_is_emergency if goal_valid > 0.0 else 0.0),
            ],
            dtype=np.float32,
        )
        if obs.shape[0] != self.obs_dim:
            raise ValueError(f"UAV obs dim mismatch: got {obs.shape[0]}, expected {self.obs_dim}")
        return obs

    def _squashed_log_prob(self, mu: Tensor, log_std: Tensor, raw_action: Tensor) -> Tensor:
        std = torch.exp(log_std).clamp(min=1e-4, max=2.0)
        dist = Normal(mu, std)
        squashed = torch.tanh(raw_action)
        logp = dist.log_prob(raw_action) - torch.log(1.0 - squashed.pow(2) + self._eps)
        return logp.sum(dim=-1)

    def evaluate_actions(
        self, obs_batch: Tensor, raw_action_batch: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            (log_prob, entropy, value)
        """
        mu, log_std, value = self.model(obs_batch)
        std = torch.exp(log_std).clamp(min=1e-4, max=2.0)
        dist = Normal(mu, std)
        squashed = torch.tanh(raw_action_batch)
        logp = dist.log_prob(raw_action_batch) - torch.log(1.0 - squashed.pow(2) + self._eps)
        logp = logp.sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return logp, entropy, value

    def act(
        self,
        env,
        high_goals: Optional[Dict[str, Optional[str]]] = None,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, object], List[Dict[str, float]]]:
        """
        Returns:
            actions, step_records_for_ppo
        """
        step_now = int(env.state.step_index)
        self._truck_policy._episode_reset_if_needed(env)
        if int(step_now) == 0:
            self._truck_away_streak = {}

        base_actions = self._truck_policy.act(env, high_goals=high_goals)
        actions: Dict[str, object] = {}
        records: List[Dict[str, float]] = []

        # Keep truck actions from topology-aware heuristic.
        for aid, s in env.state.agents.items():
            if s.kind == AgentKind.TRUCK:
                actions[aid] = base_actions.get(aid, TruckAction(stay=True))

        for aid, s in env.state.agents.items():
            if s.kind != AgentKind.UAV:
                continue
            if bool(s.crashed):
                actions[aid] = UAVAction(vx=0.0, vy=0.0)
                self._truck_away_streak.pop(str(aid), None)
                continue

            proposed_goal: Optional[str] = None
            if high_goals is not None and high_goals.get(aid, None) is not None:
                proposed_goal = str(high_goals[aid])
            else:
                g = env._effective_goals.get(str(aid), env._recommended_goals.get(str(aid), None))
                if g is not None:
                    proposed_goal = str(g)
            goal_id = self._truck_policy._resolve_uav_goal_with_lock(
                env,
                str(aid),
                proposed_goal,
                step_now=int(step_now),
            )

            goal_agent = env.state.agents.get(str(goal_id)) if goal_id is not None else None
            goal_is_truck = bool(goal_agent is not None and goal_agent.kind == AgentKind.TRUCK)
            goal_task = env.state.tasks.get(str(goal_id)) if goal_id is not None else None
            truck_soft_guidance = False
            truck_goal_xy: Optional[Tuple[float, float]] = None

            # Discrete binding/takeoff interface is still handled at interface boundary.
            if s.follow_target is not None:
                if goal_is_truck and str(s.follow_target) == str(goal_id):
                    actions[aid] = UAVAction(vx=0.0, vy=0.0)
                    self._truck_away_streak.pop(str(aid), None)
                    continue

                # Docked UAV departure gate: keep docked until reload and charge are ready.
                # This prevents "reload done but no charging" premature takeoff.
                needs_reload = bool(getattr(s, "uav_needs_reload_flag", False)) or int(getattr(s, "uav_reload_timer", 0)) > 0
                # Energy readiness is decided by the environment's complete
                # sortie (delivery plus recovery) gate below, not a fixed SOC.
                battery_ready = True
                transfer_tid = self._truck_policy._planner_transfer_target_truck(env, str(aid), goal_task)
                if needs_reload or ((not battery_ready) and transfer_tid is None):
                    actions[aid] = UAVAction(vx=0.0, vy=0.0)
                    self._truck_away_streak.pop(str(aid), None)
                    continue

                if goal_task is not None and goal_task.kind == TaskKind.EMERGENCY:
                    if hasattr(env, "is_task_serviceable_by_agent") and (not env.is_task_serviceable_by_agent(str(aid), goal_task)):
                        actions[aid] = UAVAction(vx=0.0, vy=0.0)
                        self._truck_away_streak.pop(str(aid), None)
                        continue
                    # Strict docked takeoff gate: only launch when mission-level
                    # feasibility/stability checks pass. This avoids takeoff-no-delivery
                    # oscillations from weakly-feasible or unstable goals.
                    if self._truck_policy._allow_takeoff_for_task_goal(env, str(aid), goal_task, step_now=int(step_now)):
                        actions[aid] = UAVAction(takeoff=True)
                    else:
                        transfer_tid = self._truck_policy._planner_transfer_target_truck(env, str(aid), goal_task)
                        if transfer_tid is not None and str(getattr(s, "follow_target", "")) != str(transfer_tid):
                            actions[aid] = UAVAction(takeoff=True)
                        else:
                            actions[aid] = UAVAction(vx=0.0, vy=0.0)
                    self._truck_away_streak.pop(str(aid), None)
                    continue

                if goal_is_truck:
                    # Do not detach for truck-goal while already docked.
                    # Keeping ride state avoids pointless takeoff hops that do not
                    # contribute to emergency delivery.
                    actions[aid] = UAVAction(vx=0.0, vy=0.0)
                    self._truck_away_streak.pop(str(aid), None)
                    continue

                actions[aid] = UAVAction(vx=0.0, vy=0.0)
                self._truck_away_streak.pop(str(aid), None)
                continue

            if goal_is_truck:
                cur_xy = self._agent_xy(env, aid)
                truck_xy = self._agent_xy(env, str(goal_id))
                dist = float(np.hypot(cur_xy[0] - truck_xy[0], cur_xy[1] - truck_xy[1]))
                bind_radius = float(getattr(env.cfg, "uav_bind_radius_m", 50.0))
                if dist <= bind_radius:
                    actions[aid] = UAVAction(bind_truck_id=str(goal_id))
                    self._truck_away_streak.pop(str(aid), None)
                    continue
                # Hierarchical safety gating:
                # hard override only in critical return/docking regime.
                required_batt = self._required_rth_battery(env, aid, dist)
                safety_factor = float(max(getattr(env.cfg, "rth_safety_factor", 1.0), 0.0))
                batt_critical = bool(float(s.battery) < required_batt * safety_factor)
                dock_critical = bool(dist <= (2.0 * bind_radius))
                if batt_critical or dock_critical:
                    vmax = float(max(getattr(env.cfg, "uav_max_speed_mps", 1.0), 1e-6))
                    vx, vy = self._full_speed_to(cur_xy, truck_xy, vmax=vmax)
                    actions[aid] = UAVAction(vx=vx, vy=vy)
                    self._truck_away_streak.pop(str(aid), None)
                    continue
                # Non-critical truck goal: keep policy control with soft directional guard.
                truck_soft_guidance = True
                truck_goal_xy = truck_xy

            # Safety-guided task execution:
            # far field is learnable continuous control; near/terminal field uses
            # deterministic guidance so the UAV can reliably capture and service.
            if goal_task is not None and goal_task.kind == TaskKind.EMERGENCY:
                cur_xy = self._agent_xy(env, aid)
                task_xy = env._node_xy(int(goal_task.demand_node))
                dist = float(np.hypot(cur_xy[0] - task_xy[0], cur_xy[1] - task_xy[1]))
                vmax = float(max(getattr(env.cfg, "uav_max_speed_mps", 1.0), 1e-6))
                dt_seconds = float(max(getattr(env.cfg, "dt_seconds", 1.0), 1e-6))
                delivery_radius = float(max(getattr(env.cfg, "uav_delivery_radius_m", 40.0), 1e-6))
                terminal_radius = float(
                    max(
                        getattr(env.cfg, "uav_monitor_radius_m", delivery_radius),
                        delivery_radius,
                    )
                )
                if dist <= delivery_radius:
                    actions[aid] = UAVAction(vx=0.0, vy=0.0)
                    self._truck_away_streak.pop(str(aid), None)
                    continue
                if dist <= terminal_radius:
                    vx, vy = self._command_to_target(
                        cur_xy,
                        task_xy,
                        vmax=vmax,
                        dt_seconds=dt_seconds,
                    )
                    actions[aid] = UAVAction(vx=vx, vy=vy)
                    self._truck_away_streak.pop(str(aid), None)
                    continue

            # Continuous neural control.
            obs_np = self._build_uav_obs(env, aid, goal_id)
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device).view(1, -1)
            with torch.no_grad():
                mu, log_std, value = self.model(obs_t)
                std = torch.exp(log_std).clamp(min=1e-4, max=2.0)
                if deterministic:
                    raw = mu
                else:
                    raw = mu + std * torch.randn_like(std)
                squashed = torch.tanh(raw)
                logp = self._squashed_log_prob(mu, log_std, raw)

            vmax = float(max(getattr(env.cfg, "uav_max_speed_mps", 1.0), 1e-6))
            cmd = squashed.squeeze(0).cpu().numpy()
            vx_cmd = float(np.clip(cmd[0], -1.0, 1.0) * vmax)
            vy_cmd = float(np.clip(cmd[1], -1.0, 1.0) * vmax)

            # Soft directional guard for emergency-task pursuit in far field:
            # keep learning-based control, but prevent persistent wrong-way drift.
            min_cruise = float(max(getattr(env.cfg, "uav_far_field_min_cruise_mps", 0.0), 0.0))
            if goal_task is not None and goal_task.kind == TaskKind.EMERGENCY:
                cur_xy = self._agent_xy(env, aid)
                task_xy = env._node_xy(int(goal_task.demand_node))
                min_cos = float(getattr(env.cfg, "uav_task_goal_min_cosine", -0.05))
                alpha = float(getattr(env.cfg, "uav_task_goal_guard_alpha", 0.35))
                vx_cmd, vy_cmd = self._blend_to_goal_if_wrong_way(
                    (vx_cmd, vy_cmd),
                    cur_xy,
                    task_xy,
                    vmax=vmax,
                    min_cosine=min_cos,
                    alpha=alpha,
                )
                if min_cruise > 1e-9:
                    dist_task = float(np.hypot(cur_xy[0] - task_xy[0], cur_xy[1] - task_xy[1]))
                    delivery_radius = float(max(getattr(env.cfg, "uav_delivery_radius_m", 40.0), 1e-6))
                    terminal_radius = float(max(getattr(env.cfg, "uav_monitor_radius_m", delivery_radius), delivery_radius))
                    if dist_task > terminal_radius:
                        vx_cmd, vy_cmd = self._enforce_min_cruise_speed(
                            (vx_cmd, vy_cmd),
                            cur_xy,
                            task_xy,
                            vmax=vmax,
                            min_speed=min_cruise,
                        )
            elif truck_soft_guidance and truck_goal_xy is not None:
                cur_xy = self._agent_xy(env, aid)
                min_cos = float(getattr(env.cfg, "uav_truck_goal_min_cosine", -0.10))
                alpha = float(getattr(env.cfg, "uav_truck_goal_guard_alpha", 0.15))
                vx_cmd, vy_cmd = self._blend_to_goal_if_wrong_way(
                    (vx_cmd, vy_cmd),
                    cur_xy,
                    truck_goal_xy,
                    vmax=vmax,
                    min_cosine=min_cos,
                    alpha=alpha,
                )
                cos_now = self._alignment_cos((vx_cmd, vy_cmd), cur_xy, truck_goal_xy)
                away_cos = float(getattr(env.cfg, "uav_truck_goal_away_cosine", 0.0))
                away_steps = int(max(getattr(env.cfg, "uav_truck_goal_watchdog_steps", 3), 1))
                streak = int(self._truck_away_streak.get(str(aid), 0))
                if cos_now < away_cos:
                    streak += 1
                else:
                    streak = 0
                if streak >= away_steps:
                    pulse_alpha = float(getattr(env.cfg, "uav_truck_goal_pulse_alpha", 0.65))
                    vx_cmd, vy_cmd = self._blend_to_goal_if_wrong_way(
                        (vx_cmd, vy_cmd),
                        cur_xy,
                        truck_goal_xy,
                        vmax=vmax,
                        min_cosine=1.0,
                        alpha=pulse_alpha,
                    )
                    streak = 0
                self._truck_away_streak[str(aid)] = int(streak)
                if min_cruise > 1e-9:
                    dist_truck = float(np.hypot(cur_xy[0] - truck_goal_xy[0], cur_xy[1] - truck_goal_xy[1]))
                    bind_radius = float(max(getattr(env.cfg, "uav_bind_radius_m", 50.0), 1e-6))
                    if dist_truck > 2.0 * bind_radius:
                        vx_cmd, vy_cmd = self._enforce_min_cruise_speed(
                            (vx_cmd, vy_cmd),
                            cur_xy,
                            truck_goal_xy,
                            vmax=vmax,
                            min_speed=min_cruise,
                        )
            else:
                self._truck_away_streak.pop(str(aid), None)
            actions[aid] = UAVAction(vx=vx_cmd, vy=vy_cmd)
            records.append(
                {
                    "aid": str(aid),
                    "obs": obs_np.tolist(),
                    "raw_action": raw.squeeze(0).cpu().numpy().tolist(),
                    "old_logp": float(logp.item()),
                    "old_value": float(value.item()),
                    "reward": 0.0,
                    "done": 0.0,
                    "return": 0.0,
                }
            )

            # Optional safety: UAV should not be assigned to normal tasks.
            if goal_task is not None and goal_task.kind != TaskKind.EMERGENCY:
                actions[aid] = UAVAction(vx=0.0, vy=0.0)

        return actions, records





