from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from hetgat_hrl.agents.hetgat_risk import RiskMaskedHetGAT
from hetgat_hrl.core.mdp_spec import AgentKind, TaskKind, TaskStatus


@dataclass
class HRLPlannerState:
    step_last_refresh: int = 0
    goals: Dict[str, Optional[str]] = field(default_factory=dict)
    resolved_tasks_last: int = 0
    goal_assigned_step: Dict[str, int] = field(default_factory=dict)
    risk_spike_latch: bool = False
    # Edge-trigger memory: avoid repeated arrival-triggered replans while
    # agent remains at the same reached goal.
    goal_reached_latch: Dict[str, str] = field(default_factory=dict)


class NeuralGoalAllocator(nn.Module):
    """
    Small neural scorer for high-level task assignment.
    Input features per (agent, task): [dist_norm, emergency_flag].
    Output: scalar preference score.
    """

    def __init__(self, in_dim: int = 2, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        # Deterministic initializer to prefer near + emergency before training.
        with torch.no_grad():
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [N,2] -> score [N]
        return self.net(feat).squeeze(-1)


class RiskTriggeredHRLPlanner:
    """
    HRL high-level planner with unique authority on task recommendation.
    Refresh policy:
    - every decision_interval steps
    - on risk spike
    - on task completion/failure count change
    """

    def __init__(self, decision_interval: int = 5, seed: int = 0, encoder_type: str = "hetgat"):
        self.decision_interval = max(int(decision_interval), 1)
        self.state = HRLPlannerState()
        self.encoder_type = str(encoder_type).strip().lower()
        if self.encoder_type not in {"hetgat", "mlp", "pooled"}:
            raise ValueError(f"unsupported encoder_type={encoder_type!r}, expected hetgat|mlp|pooled")
        torch.manual_seed(seed)
        self.allocator = NeuralGoalAllocator(in_dim=2, hidden_dim=16)
        self.risk_gat = RiskMaskedHetGAT(in_dim=4, hidden_dim=16, beta=1.0)
        self._last_effective_risk_beta: float = float(self.risk_gat.beta)
        self.hetgat_projector = nn.Sequential(
            nn.Linear(2 + 16, 16),
            nn.Tanh(),
            nn.Linear(16, 2),
        )
        self.mlp_encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 2),
        )
        self.last_replan_reason: str = "init"
        self._last_refresh_flags: Dict[str, bool] = {}
        # Robust shared-map refresh tracking across planner/env timing.
        self._last_seen_shared_map_update_count_total: int = 0

    def _effective_risk_beta(self, env) -> float:
        cfg_beta = float(max(getattr(env.cfg, "risk_gat_beta", 1.0), 0.0))
        # Hard degrade path: disabling RTH/risk mask must also disable risk attention masking.
        if not bool(getattr(env.cfg, "enable_rth_mask", True)):
            return 0.0
        return float(cfg_beta)

    def _sync_risk_beta(self, env) -> None:
        eff = float(self._effective_risk_beta(env))
        self.risk_gat.beta = float(eff)
        self._last_effective_risk_beta = float(eff)

    def high_level_parameters(self, encoder_type: Optional[str] = None) -> List[nn.Parameter]:
        et = str(encoder_type or self.encoder_type).strip().lower()
        params: List[nn.Parameter] = list(self.allocator.parameters())
        if et == "hetgat":
            params += list(self.risk_gat.parameters())
            params += list(self.hetgat_projector.parameters())
        elif et == "mlp":
            params += list(self.mlp_encoder.parameters())
        return params

    def capture_graph_snapshot(self, env) -> Optional[Dict[str, torch.Tensor]]:
        if self.encoder_type != "hetgat":
            return None
        self._sync_risk_beta(env)
        n = len(env.topology.nodes)
        if n <= 0:
            return None
        x_rows = []
        for i in range(n):
            node = env.topology.nodes[i]
            hz = env.hazards.weather_at((float(node.x), float(node.y)))
            x_rows.append(
                [
                    float(hz.rain / max(env.cfg.base_rainfall_mmh, 1e-6)),
                    float(hz.wind / max(env.cfg.base_wind_mps, 1e-6)),
                    float(hz.quake),
                    float(node.slope_norm),
                ]
            )
        edges = []
        risks = []
        for src, nbs in env.topology.adjacency.items():
            for dst in nbs:
                if src == dst:
                    continue
                edges.append((int(src), int(dst)))
                k = (min(int(src), int(dst)), max(int(src), int(dst)))
                risks.append(float(env.hazards.last_edge_pstep.get(k, 0.0)))
        if not edges:
            return None
        return {
            "node_x": torch.tensor(x_rows, dtype=torch.float32),
            "edge_index": torch.tensor(edges, dtype=torch.long).t().contiguous(),
            "edge_risk": torch.tensor(risks, dtype=torch.float32),
        }

    def encode_candidates(
        self,
        base_feat: torch.Tensor,
        task_node_idx: List[int],
        encoder_type: Optional[str] = None,
        graph_snapshot: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Encode per-candidate features while preserving candidate discrimination.
        Returns encoded candidate tensor with shape [N, 2] for allocator head.
        """
        et = str(encoder_type or self.encoder_type).strip().lower()
        x = base_feat.to(torch.float32)
        if et == "pooled":
            pooled = x.mean(dim=0, keepdim=True)
            return pooled.repeat(x.shape[0], 1)
        if et == "mlp":
            return self.mlp_encoder(x)
        if et != "hetgat":
            raise ValueError(f"unsupported encoder_type={et}")

        if graph_snapshot is None:
            z = torch.zeros((x.shape[0], 16), dtype=x.dtype, device=x.device)
            return self.hetgat_projector(torch.cat([x, z], dim=-1))

        node_x = graph_snapshot["node_x"].to(dtype=x.dtype, device=x.device)
        edge_index = graph_snapshot["edge_index"].to(device=x.device)
        edge_risk = graph_snapshot["edge_risk"].to(dtype=x.dtype, device=x.device)
        emb, _ = self.risk_gat(node_x, edge_index, edge_risk=edge_risk)  # [N,16]

        idx = torch.tensor(task_node_idx, dtype=torch.long, device=x.device)
        idx = torch.clamp(idx, min=0, max=max(int(emb.shape[0]) - 1, 0))
        task_emb = emb[idx] if emb.numel() > 0 else torch.zeros((x.shape[0], 16), dtype=x.dtype, device=x.device)
        return self.hetgat_projector(torch.cat([x, task_emb], dim=-1))

    def _resolved_count(self, env) -> int:
        return int(
            sum(
                1
                for t in env.state.tasks.values()
                if t.status in (TaskStatus.DELIVERED, TaskStatus.FAILED)
            )
        )

    def _truck_dead_end_refresh(self, env) -> bool:
        """
        Trigger replan when a truck with an active task-goal reaches a local dead-end
        (only one legal outgoing neighbor), so upper layer can re-route globally.
        """
        legal = env.legal_actions() if hasattr(env, "legal_actions") else {}
        for aid, tid in self.state.goals.items():
            if tid is None:
                continue
            a = env.state.agents.get(str(aid), None)
            if a is None or a.kind != AgentKind.TRUCK or bool(a.crashed):
                continue
            t = env.state.tasks.get(str(tid), None)
            if t is None or t.status != TaskStatus.PENDING:
                continue
            if a.node is not None and int(a.node) == int(t.demand_node):
                continue
            nbs = legal.get(str(aid), {}).get("neighbors", [])
            if len(nbs) <= 1:
                return True
        return False

    def _uav_emergency_refresh(self, env) -> bool:
        """
        Trigger replan if a UAV is still pursuing a task while battery is already
        in emergency zone. This enables urgent switch to return-to-truck plans.
        """
        thr = float(max(getattr(env.cfg, "uav_replan_emergency_battery_threshold", 0.30), 0.0))
        for aid, tid in self.state.goals.items():
            if tid is None:
                continue
            a = env.state.agents.get(str(aid), None)
            if a is None or a.kind != AgentKind.UAV or bool(a.crashed):
                continue
            goal_agent = env.state.agents.get(str(tid), None)
            if goal_agent is not None and goal_agent.kind == AgentKind.TRUCK:
                continue
            goal_task = env.state.tasks.get(str(tid), None)
            if goal_task is None or goal_task.status != TaskStatus.PENDING:
                continue
            if float(a.battery) < thr:
                return True
        return False

    def _should_refresh(self, env) -> bool:
        step_now = int(env.state.step_index)
        if step_now == 0 and int(self.state.step_last_refresh) > 0:
            self.state.goals = {}
            self.state.goal_assigned_step = {}
            self.state.resolved_tasks_last = 0
            self.state.risk_spike_latch = False
            self.state.goal_reached_latch = {}
            self._last_seen_shared_map_update_count_total = int(
                max(getattr(env, "_shared_map_update_count_total", 0), 0)
            )

        since_last = int(step_now - self.state.step_last_refresh)
        by_interval = (since_last >= self.decision_interval)
        risk_now = bool(env.state.hazard.risk_spike)
        edge_only = bool(getattr(env.cfg, "hrl_risk_spike_edge_trigger_only", True))
        by_risk = bool(risk_now and ((not edge_only) or (not bool(self.state.risk_spike_latch))))
        self.state.risk_spike_latch = bool(risk_now)
        resolved_now = self._resolved_count(env)
        by_resolution = resolved_now != self.state.resolved_tasks_last
        by_arrival = False
        reached_latch = self.state.goal_reached_latch
        # Clear stale latches when current goal changes/removes.
        for latch_aid, latch_goal in list(reached_latch.items()):
            cur_goal = self.state.goals.get(str(latch_aid), None)
            if cur_goal is None or str(cur_goal) != str(latch_goal):
                reached_latch.pop(str(latch_aid), None)

        for aid, tid in self.state.goals.items():
            if tid is None:
                continue
            aid = str(aid)
            tid = str(tid)
            a = env.state.agents.get(aid, None)
            if a is None:
                continue

            reached_now = False
            t = env.state.tasks.get(tid)
            # Virtual truck goal should not be treated as "task resolved".
            # It is considered reached only on first attach/entering bind window.
            if t is None:
                goal_agent = env.state.agents.get(tid, None)
                if (
                    a.kind == AgentKind.UAV
                    and goal_agent is not None
                    and goal_agent.kind == AgentKind.TRUCK
                ):
                    if a.follow_target is not None and str(a.follow_target) == tid:
                        reached_now = True
                    else:
                        ax, ay = self._agent_xy(env, aid)
                        tx, ty = self._agent_xy(env, tid)
                        d = float(((ax - tx) ** 2 + (ay - ty) ** 2) ** 0.5)
                        bind_window = float(getattr(env.cfg, "uav_bind_radius_m", 50.0))
                        if hasattr(env, "_uav_bind_window_m"):
                            try:
                                bind_window = float(env._uav_bind_window_m(goal_agent))
                            except Exception:
                                bind_window = float(getattr(env.cfg, "uav_bind_radius_m", 50.0))
                        if d <= bind_window:
                            reached_now = True
                else:
                    # Unknown goal id: force one refresh to recover mapping.
                    by_arrival = True
                    reached_latch.pop(aid, None)
                    break
            else:
                if t.status in (TaskStatus.DELIVERED, TaskStatus.FAILED):
                    reached_now = True
                elif a.kind == AgentKind.TRUCK and a.node is not None:
                    if int(a.node) == int(t.demand_node):
                        reached_now = True
                elif a.kind == AgentKind.UAV and a.pos_xy is not None:
                    nx = env.topology.nodes[int(t.demand_node)]
                    d = float(((a.pos_xy[0] - nx.x) ** 2 + (a.pos_xy[1] - nx.y) ** 2) ** 0.5)
                    if d <= float(env.cfg.uav_delivery_radius_m):
                        reached_now = True

            prev_reached_goal = reached_latch.get(aid, None)
            if reached_now:
                if str(prev_reached_goal) != tid:
                    by_arrival = True
                reached_latch[aid] = tid
                if by_arrival:
                    break
            else:
                if str(prev_reached_goal) == tid:
                    reached_latch.pop(aid, None)
        by_truck_dead_end = self._truck_dead_end_refresh(env)
        by_uav_emergency = self._uav_emergency_refresh(env)
        shared_map_update_count_now = int(max(getattr(env, "_shared_map_update_count_total", 0), 0))
        shared_map_update_event = bool(getattr(env, "_shared_map_update_event_step", False))
        shared_map_update_delta = bool(
            shared_map_update_count_now > int(self._last_seen_shared_map_update_count_total)
        )
        by_map_update = bool(
            getattr(env.cfg, "road_shared_replan_on_update", True)
            and (shared_map_update_event or shared_map_update_delta)
        )
        # Cooldown to suppress non-critical thrashing:
        # still allow urgent refreshes (risk spike / dead-end / emergency / map-update / resolution).
        cooldown = int(max(getattr(env.cfg, "hrl_replan_cooldown_steps", 3), 0))
        if since_last < cooldown and bool(self.state.goals):
            do_refresh = bool(by_risk or by_truck_dead_end or by_uav_emergency or by_map_update or by_resolution)
        else:
            do_refresh = bool(
                by_interval
                or by_risk
                or by_resolution
                or by_arrival
                or by_truck_dead_end
                or by_uav_emergency
                or by_map_update
                or not self.state.goals
            )
        if by_map_update and hasattr(env, "_shared_map_update_event_step"):
            env._shared_map_update_event_step = False
        self._last_refresh_flags = {
            "interval": bool(by_interval),
            "risk_spike": bool(by_risk),
            "resolution": bool(by_resolution),
            "arrival": bool(by_arrival),
            "truck_dead_end": bool(by_truck_dead_end),
            "uav_emergency": bool(by_uav_emergency),
            "map_update": bool(by_map_update),
            "empty_goals": bool(not self.state.goals),
            "refresh": bool(do_refresh),
        }
        self._last_seen_shared_map_update_count_total = int(
            max(self._last_seen_shared_map_update_count_total, shared_map_update_count_now)
        )
        return bool(do_refresh)

    def _truck_task_reachable(self, env, aid: str, task) -> bool:
        st = env.state.agents.get(str(aid), None)
        if st is None or st.kind != AgentKind.TRUCK or st.node is None:
            return False
        if task is None or task.status != TaskStatus.PENDING:
            return False
        src = int(st.node)
        dst = int(task.demand_node)
        if hasattr(env, "_decision_shortest_path_distance"):
            d = float(env._decision_shortest_path_distance(src, dst))
        else:
            d = float(env.topology.shortest_path_distance(src, dst, ignore_blocked=False))
        return bool(np.isfinite(d))

    def _nearest_truck_reachable_task(
        self,
        env,
        aid: str,
        *,
        used_tasks: Optional[set] = None,
        allow_used: bool = False,
    ) -> Optional[str]:
        st = env.state.agents.get(str(aid), None)
        if st is None or st.kind != AgentKind.TRUCK or bool(getattr(st, "crashed", False)):
            return None
        blocked = set(used_tasks or set())
        best_tid: Optional[str] = None
        best_dist = float("inf")
        best_priority = 10
        pending_normal_exists = any(
            t.status == TaskStatus.PENDING and t.kind == TaskKind.NORMAL
            for t in env.state.tasks.values()
        )
        for t in env.state.tasks.values():
            if t.status != TaskStatus.PENDING:
                continue
            tid = str(t.task_id)
            if (not allow_used) and tid in blocked:
                continue
            if hasattr(env, "is_task_serviceable_by_agent") and (
                not bool(env.is_task_serviceable_by_agent(str(aid), t))
            ):
                continue
            if not self._truck_task_reachable(env, str(aid), t):
                continue
            if pending_normal_exists:
                priority = 0 if t.kind == TaskKind.NORMAL else 1
            else:
                priority = 0
            d = float(env._agent_distance_to_task(str(aid), t))
            if (priority < best_priority) or (
                priority == best_priority and d + 1e-9 < best_dist
            ):
                best_priority = int(priority)
                best_dist = float(d)
                best_tid = tid
        return best_tid

    def _candidate_tasks(self, env, aid: str) -> List[object]:
        a = env.state.agents[aid]
        tasks = [
            t
            for t in env.state.tasks.values()
            if t.status == TaskStatus.PENDING
        ]
        if a.kind == AgentKind.UAV:
            tasks = [t for t in tasks if t.kind == TaskKind.EMERGENCY]
        if hasattr(env, "is_task_serviceable_by_agent"):
            tasks = [t for t in tasks if bool(env.is_task_serviceable_by_agent(str(aid), t))]
        if a.kind == AgentKind.TRUCK:
            tasks = [t for t in tasks if self._truck_task_reachable(env, str(aid), t)]
        return tasks

    def _ordered_agents(self, env) -> List[str]:
        return sorted(
            env.state.agents.keys(),
            key=lambda aid: 0 if env.state.agents[aid].kind == AgentKind.UAV else 1,
        )

    def _agent_xy(self, env, aid: str) -> Tuple[float, float]:
        st = env.state.agents[aid]
        if st.pos_xy is not None:
            return float(st.pos_xy[0]), float(st.pos_xy[1])
        return env._node_xy(int(st.node or 0))

    def _nearest_truck(self, env, aid: str) -> Tuple[Optional[str], float]:
        ax, ay = self._agent_xy(env, aid)
        best_id: Optional[str] = None
        best_d = float("inf")
        for tid, ts in env.state.agents.items():
            if ts.kind != AgentKind.TRUCK:
                continue
            tx, ty = self._agent_xy(env, str(tid))
            d = float(((ax - tx) ** 2 + (ay - ty) ** 2) ** 0.5)
            if d < best_d:
                best_d = d
                best_id = str(tid)
        return best_id, best_d

    def _required_rth_battery(self, env, aid: str, dist_to_truck: float) -> float:
        st = env.state.agents[aid]
        base_discharge_per_m = float(
            max(getattr(env.cfg, "uav_flight_discharge_per_m", 1e-6), 1e-6)
        )
        headwind_coeff = float(max(getattr(env.cfg, "uav_headwind_energy_coeff", 0.04), 0.0))
        rain_coeff = float(max(getattr(env.cfg, "uav_rain_energy_coeff", 0.02), 0.0))
        base_wind = float(max(getattr(env.cfg, "base_wind_mps", 0.0), 0.0))
        base_rain = float(max(getattr(env.cfg, "base_rainfall_mmh", 0.0), 0.0))
        cargo_unit_kg = float(max(getattr(env.cfg, "cargo_unit_kg", 200.0), 1e-6))
        m_load_kg = float(max(getattr(st, "cargo", 0.0), 0.0)) * cargo_unit_kg
        load_factor = 1.0 + 0.018 * m_load_kg
        weather_factor = 1.0 + headwind_coeff * base_wind + rain_coeff * base_rain
        safe_discharge_rate = base_discharge_per_m * weather_factor * load_factor
        return float(max(0.0, dist_to_truck) * safe_discharge_rate)

    def _nearest_truck_from_xy(self, env, xy: Tuple[float, float]) -> Tuple[Optional[str], float]:
        x, y = float(xy[0]), float(xy[1])
        best_id: Optional[str] = None
        best_d = float("inf")
        for tid, ts in env.state.agents.items():
            if ts.kind != AgentKind.TRUCK:
                continue
            tx, ty = self._agent_xy(env, str(tid))
            d = float(((x - tx) ** 2 + (y - ty) ** 2) ** 0.5)
            if d < best_d:
                best_d = d
                best_id = str(tid)
        return best_id, best_d

    def _uav_goal_valid(self, env, aid: str, goal_id: Optional[str]) -> bool:
        if goal_id is None:
            return False
        gid = str(goal_id)
        ag = env.state.agents.get(gid, None)
        if ag is not None:
            return bool(ag.kind == AgentKind.TRUCK and (not bool(getattr(ag, "crashed", False))))
        task = env.state.tasks.get(gid, None)
        if task is None:
            return False
        if task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
            return False
        if hasattr(env, "is_task_serviceable_by_agent"):
            return bool(env.is_task_serviceable_by_agent(str(aid), task))
        return True

    def _uav_goal_distance(self, env, aid: str, goal_id: Optional[str]) -> float:
        if goal_id is None:
            return float("inf")
        gid = str(goal_id)
        ag = env.state.agents.get(gid, None)
        if ag is not None and ag.kind == AgentKind.TRUCK:
            ax, ay = self._agent_xy(env, str(aid))
            tx, ty = self._agent_xy(env, gid)
            return float(((ax - tx) ** 2 + (ay - ty) ** 2) ** 0.5)
        task = env.state.tasks.get(gid, None)
        if task is not None and task.kind == TaskKind.EMERGENCY and task.status == TaskStatus.PENDING:
            return float(env._agent_distance_to_task(str(aid), task))
        return float("inf")

    def _uav_recovery_urgent(self, env, aid: str) -> bool:
        st = env.state.agents.get(str(aid), None)
        if st is None or st.kind != AgentKind.UAV or bool(getattr(st, "crashed", False)):
            return False
        if hasattr(env, "_uav_recovery_required"):
            try:
                if bool(env._uav_recovery_required(str(aid))):
                    return True
            except Exception:
                pass
        force_thr = float(np.clip(getattr(env.cfg, "uav_low_battery_force_recover_threshold", 0.25), 0.0, 1.0))
        if float(getattr(st, "battery", 0.0)) <= force_thr:
            return True
        if bool(getattr(st, "uav_needs_reload_flag", False)):
            return True
        if hasattr(env, "_uav_loaded"):
            try:
                if not bool(env._uav_loaded(str(aid))):
                    return True
            except Exception:
                pass
        if bool(getattr(env, "_uav_forced_rth_latch", {}).get(str(aid), False)):
            return True
        return False


    def _goal_valid_for_agent(self, env, aid: str, goal_id: Optional[str]) -> bool:
        if goal_id is None:
            return False
        a = env.state.agents.get(str(aid), None)
        if a is None:
            return False
        gid = str(goal_id)
        t = env.state.tasks.get(gid, None)
        if t is not None:
            if t.status != TaskStatus.PENDING:
                return False
            if hasattr(env, "is_task_serviceable_by_agent"):
                return bool(env.is_task_serviceable_by_agent(str(aid), t))
            return True
        ag = env.state.agents.get(gid, None)
        if a.kind == AgentKind.UAV and ag is not None and ag.kind == AgentKind.TRUCK:
            return bool(not bool(getattr(ag, "crashed", False)))
        return False

    def _goal_min_hold_steps(self, env, aid: str) -> int:
        a = env.state.agents.get(str(aid), None)
        base = int(max(getattr(env.cfg, "hrl_goal_min_hold_steps", 12), 0))
        if a is not None and a.kind == AgentKind.UAV:
            base = max(base, int(max(getattr(env.cfg, "uav_goal_lock_steps", 0), 0)))
        return int(base)

    def _goal_switch_margin(self, env, aid: str) -> float:
        a = env.state.agents.get(str(aid), None)
        if a is not None and a.kind == AgentKind.UAV:
            return float(max(getattr(env.cfg, "hrl_uav_goal_switch_margin", getattr(env.cfg, "hrl_goal_switch_margin", 0.20)), 0.0))
        return float(max(getattr(env.cfg, "hrl_truck_goal_switch_margin", getattr(env.cfg, "hrl_goal_switch_margin", 0.20)), 0.0))

    def _stabilize_goal_choice(
        self,
        env,
        aid: str,
        candidate_ids: List[str],
        logits: torch.Tensor,
        proposed_goal: Optional[str],
        used_tasks: set,
    ) -> Optional[str]:
        if proposed_goal is None:
            return None
        cand = [str(x) for x in candidate_ids]
        cand_set = set(cand)
        goal = str(proposed_goal)
        prev_goal = self.state.goals.get(str(aid), None)
        if prev_goal is None:
            return goal
        prev_goal = str(prev_goal)
        if prev_goal not in cand_set:
            return goal
        if not self._goal_valid_for_agent(env, str(aid), prev_goal):
            return goal

        # Safety override: when UAV is in urgent recovery, allow switching to truck goal.
        st = env.state.agents.get(str(aid), None)
        if st is not None and st.kind == AgentKind.UAV and self._uav_recovery_urgent(env, str(aid)):
            prev_ag = env.state.agents.get(str(prev_goal), None)
            goal_ag = env.state.agents.get(str(goal), None)
            if (goal_ag is not None and goal_ag.kind == AgentKind.TRUCK) and not (prev_ag is not None and prev_ag.kind == AgentKind.TRUCK):
                return goal

        # Respect minimum hold time while previous goal is still valid.
        step_now = int(env.state.step_index)
        assigned = int(self.state.goal_assigned_step.get(str(aid), step_now))
        hold_steps = int(self._goal_min_hold_steps(env, str(aid)))
        elapsed = int(step_now - assigned)
        prev_task = env.state.tasks.get(str(prev_goal), None)
        prev_busy = bool(prev_task is not None and str(prev_task.task_id) in used_tasks)
        if elapsed < hold_steps and (not prev_busy):
            return prev_goal

        if str(goal) == str(prev_goal):
            return goal

        # Hysteresis by score margin: switch only if materially better than previous.
        prev_idx = -1
        new_idx = -1
        for i, tid in enumerate(cand):
            if str(tid) == str(prev_goal):
                prev_idx = int(i)
            if str(tid) == str(goal):
                new_idx = int(i)
        if prev_idx < 0 or new_idx < 0:
            return goal
        if prev_busy:
            return goal
        margin = float(self._goal_switch_margin(env, str(aid)))
        prev_score = float(logits[prev_idx].item())
        new_score = float(logits[new_idx].item())
        if new_score <= float(prev_score + margin):
            return prev_goal
        return goal

    def _resolve_uav_goal_with_lock(
        self,
        env,
        aid: str,
        proposed_goal: Optional[str],
        candidate_ids: List[str],
        used_tasks: set,
    ) -> Optional[str]:
        goal = None if proposed_goal is None else str(proposed_goal)
        st = env.state.agents.get(str(aid), None)
        if st is None or st.kind != AgentKind.UAV:
            return goal

        cand_ids = [str(x) for x in candidate_ids]
        cand_set = set(cand_ids)
        prev_goal = self.state.goals.get(str(aid), None)
        step_now = int(env.state.step_index)
        assigned_step = int(self.state.goal_assigned_step.get(str(aid), step_now))
        lock_steps = int(max(getattr(env.cfg, "uav_goal_lock_steps", 0), 0))

        truck_candidates = [
            gid for gid in cand_ids
            if (env.state.agents.get(str(gid), None) is not None)
            and env.state.agents[str(gid)].kind == AgentKind.TRUCK
            and (not bool(getattr(env.state.agents[str(gid)], "crashed", False)))
        ]

        if self._uav_recovery_urgent(env, str(aid)):
            # Docked anti-oscillation: when UAV is already on truck, keep a valid
            # emergency mission goal if available instead of bouncing task<->truck.
            if st.follow_target is not None:
                if prev_goal is not None and str(prev_goal) in cand_set:
                    prev_task = env.state.tasks.get(str(prev_goal), None)
                    if (
                        prev_task is not None
                        and prev_task.kind == TaskKind.EMERGENCY
                        and prev_task.status == TaskStatus.PENDING
                        and self._uav_goal_valid(env, str(aid), str(prev_goal))
                    ):
                        return str(prev_goal)
                if goal is not None and str(goal) in cand_set:
                    goal_task = env.state.tasks.get(str(goal), None)
                    if (
                        goal_task is not None
                        and goal_task.kind == TaskKind.EMERGENCY
                        and goal_task.status == TaskStatus.PENDING
                        and self._uav_goal_valid(env, str(aid), str(goal))
                    ):
                        return str(goal)

            if prev_goal is not None:
                prev_agent = env.state.agents.get(str(prev_goal), None)
                if (
                    prev_agent is not None
                    and prev_agent.kind == AgentKind.TRUCK
                    and (not bool(getattr(prev_agent, "crashed", False)))
                    and str(prev_goal) in cand_set
                ):
                    return str(prev_goal)
            if truck_candidates:
                ax, ay = self._agent_xy(env, str(aid))
                return min(
                    truck_candidates,
                    key=lambda tid: float(
                        ((ax - self._agent_xy(env, str(tid))[0]) ** 2 + (ay - self._agent_xy(env, str(tid))[1]) ** 2) ** 0.5
                    ),
                )
            return goal

        if (
            lock_steps > 0
            and prev_goal is not None
            and goal is not None
            and str(goal) != str(prev_goal)
            and (step_now - assigned_step) < lock_steps
            and self._uav_goal_valid(env, str(aid), str(prev_goal))
            and str(prev_goal) in cand_set
        ):
            prev_task = env.state.tasks.get(str(prev_goal), None)
            if not (prev_task is not None and str(prev_task.task_id) in used_tasks):
                goal = str(prev_goal)

        # Docked truck<->task anti-bounce: while riding a truck and not in urgent
        # recovery, avoid switching between truck-id and emergency-task-id unless
        # previous goal is no longer valid.
        if (
            st.follow_target is not None
            and prev_goal is not None
            and goal is not None
            and str(goal) != str(prev_goal)
            and (not self._uav_recovery_urgent(env, str(aid)))
            and str(prev_goal) in cand_set
            and self._uav_goal_valid(env, str(aid), str(prev_goal))
        ):
            prev_task = env.state.tasks.get(str(prev_goal), None)
            goal_task = env.state.tasks.get(str(goal), None)
            prev_agent = env.state.agents.get(str(prev_goal), None)
            goal_agent = env.state.agents.get(str(goal), None)
            prev_is_task = bool(prev_task is not None and prev_task.kind == TaskKind.EMERGENCY and prev_task.status == TaskStatus.PENDING)
            goal_is_task = bool(goal_task is not None and goal_task.kind == TaskKind.EMERGENCY and goal_task.status == TaskStatus.PENDING)
            prev_is_truck = bool(prev_agent is not None and prev_agent.kind == AgentKind.TRUCK)
            goal_is_truck = bool(goal_agent is not None and goal_agent.kind == AgentKind.TRUCK)
            if (prev_is_task and goal_is_truck) or (prev_is_truck and goal_is_task):
                goal = str(prev_goal)

        batt = float(getattr(st, "battery", 0.0))
        low_lock_thr = float(np.clip(getattr(env.cfg, "uav_low_battery_goal_lock_threshold", 0.35), 0.0, 1.0))
        if (
            batt < low_lock_thr
            and prev_goal is not None
            and goal is not None
            and str(goal) != str(prev_goal)
            and self._uav_goal_valid(env, str(aid), str(prev_goal))
            and str(prev_goal) in cand_set
        ):
            prev_d = float(self._uav_goal_distance(env, str(aid), str(prev_goal)))
            new_d = float(self._uav_goal_distance(env, str(aid), str(goal)))
            prev_task = env.state.tasks.get(str(prev_goal), None)
            prev_busy = bool(prev_task is not None and str(prev_task.task_id) in used_tasks)
            if (not prev_busy) and np.isfinite(prev_d) and np.isfinite(new_d) and new_d > prev_d + 1e-6:
                goal = str(prev_goal)

        # Hard docked far-switch guard:
        # if UAV is already riding truck and both old/new goals are emergency tasks,
        # do not switch to a significantly farther emergency unless urgent recovery.
        if (
            st.follow_target is not None
            and prev_goal is not None
            and goal is not None
            and str(goal) != str(prev_goal)
            and (not self._uav_recovery_urgent(env, str(aid)))
        ):
            prev_task = env.state.tasks.get(str(prev_goal), None)
            new_task = env.state.tasks.get(str(goal), None)
            if (
                prev_task is not None
                and new_task is not None
                and prev_task.kind == TaskKind.EMERGENCY
                and new_task.kind == TaskKind.EMERGENCY
                and prev_task.status == TaskStatus.PENDING
                and new_task.status == TaskStatus.PENDING
            ):
                prev_d = float(self._uav_goal_distance(env, str(aid), str(prev_goal)))
                new_d = float(self._uav_goal_distance(env, str(aid), str(goal)))
                margin = float(max(getattr(env.cfg, "uav_docked_hard_far_switch_margin_m", 60.0), 0.0))
                if np.isfinite(prev_d) and np.isfinite(new_d) and new_d > (prev_d + margin):
                    goal = str(prev_goal)

        # Docked nearest-task correction:
        # if UAV is on truck and current emergency goal is materially farther than
        # nearest serviceable emergency candidate, force correction to nearest.
        if (
            st.follow_target is not None
            and bool(getattr(env.cfg, "uav_docked_task_shortlist_enabled", True))
            and (not self._uav_recovery_urgent(env, str(aid)))
        ):
            em_cands: List[str] = []
            for tid in cand_ids:
                task = env.state.tasks.get(str(tid), None)
                if task is not None and task.kind == TaskKind.EMERGENCY and task.status == TaskStatus.PENDING:
                    em_cands.append(str(tid))
            if em_cands:
                nearest_tid = min(em_cands, key=lambda tid: float(self._uav_goal_distance(env, str(aid), str(tid))))
                nearest_d = float(self._uav_goal_distance(env, str(aid), str(nearest_tid)))
                cur_task = env.state.tasks.get(str(goal), None) if goal is not None else None
                margin = float(max(getattr(env.cfg, "uav_docked_hard_far_switch_margin_m", 60.0), 0.0))
                if cur_task is None or cur_task.kind != TaskKind.EMERGENCY or cur_task.status != TaskStatus.PENDING:
                    goal = str(nearest_tid)
                else:
                    cur_d = float(self._uav_goal_distance(env, str(aid), str(goal)))
                    if np.isfinite(cur_d) and np.isfinite(nearest_d) and cur_d > (nearest_d + margin):
                        goal = str(nearest_tid)

        return goal

    def _uav_reachability_soft_bias(self, env, aid: str, task) -> float:
        """
        Soft bias only (no hard filter): prioritize emergency tasks that can
        likely be completed and safely recovered from with current battery.
        """
        st = env.state.agents[aid]
        if st.kind != AgentKind.UAV or bool(st.crashed):
            return 0.0
        if task.kind != TaskKind.EMERGENCY:
            return 0.0
        d_go = float(env._agent_distance_to_task(aid, task))
        n = env.topology.nodes[int(task.demand_node)]
        _, d_back = self._nearest_truck_from_xy(env, (float(n.x), float(n.y)))
        if not np.isfinite(d_back):
            d_back = 0.0
        recovery_buf = float(max(getattr(env.cfg, "uav_recovery_distance_buffer_m", 600.0), 0.0))
        req = self._required_rth_battery(env, aid, d_go + d_back + recovery_buf)
        safety_factor = float(max(getattr(env.cfg, "rth_safety_factor", 1.0), 0.0))
        ratio = float((req * safety_factor) / max(float(st.battery), 1e-6))
        # ratio <= 1 is feasible with margin -> positive/neutral bias
        # ratio > 1 increasingly infeasible -> negative bias
        bias = 0.15 - 0.25 * ratio
        return float(np.clip(bias, -0.35, 0.10))

    def build_candidates_for_agent(
        self,
        env,
        aid: str,
        used_tasks: Optional[set] = None,
        enable_rth_mask: Optional[bool] = None,
    ) -> Tuple[List[str], List[List[float]], List[int]]:
        used = used_tasks if used_tasks is not None else set()
        enable_mask = bool(
            getattr(env.cfg, "enable_rth_mask", True) if enable_rth_mask is None else enable_rth_mask
        )
        st = env.state.agents[aid]

        tids: List[str] = []
        feats: List[List[float]] = []
        task_nodes: List[int] = []

        cands = self._candidate_tasks(env, aid)
        # Docked mission-memory repair: keep previous pending emergency goal
        # visible in candidate set even during transient reload/recovery gating.
        if st.kind == AgentKind.UAV and (not bool(st.crashed)) and st.follow_target is not None:
            prev_gid = self.state.goals.get(str(aid), None)
            prev_task = env.state.tasks.get(str(prev_gid), None) if prev_gid is not None else None
            if (
                prev_task is not None
                and prev_task.kind == TaskKind.EMERGENCY
                and prev_task.status == TaskStatus.PENDING
                and all(str(getattr(x, 'task_id', '')) != str(prev_task.task_id) for x in cands)
            ):
                cands = [prev_task] + list(cands)

        if (
            st.kind == AgentKind.UAV
            and (not bool(st.crashed))
            and st.follow_target is not None
            and bool(getattr(env.cfg, "uav_docked_task_shortlist_enabled", True))
            and (not self._uav_recovery_urgent(env, str(aid)))
            and (not bool(getattr(st, "uav_needs_reload_flag", False)))
            and float(getattr(st, "cargo", 0.0)) > 0.0
        ):
            radius_m = float(max(getattr(env.cfg, "uav_docked_task_shortlist_radius_m", 1500.0), 0.0))
            topk = int(max(getattr(env.cfg, "uav_docked_task_shortlist_topk", 3), 1))
            ranked: List[Tuple[float, object]] = []
            for t in cands:
                if t.kind != TaskKind.EMERGENCY or t.status != TaskStatus.PENDING:
                    continue
                d = float(env._agent_distance_to_task(aid, t))
                if np.isfinite(d) and d <= radius_m:
                    ranked.append((d, t))
            if ranked:
                ranked.sort(key=lambda x: float(x[0]))
                prev_goal = self.state.goals.get(str(aid), None)
                prev_task = env.state.tasks.get(str(prev_goal), None) if prev_goal is not None else None
                prev_keep_task = None
                if prev_task is not None and prev_task.kind == TaskKind.EMERGENCY and prev_task.status == TaskStatus.PENDING:
                    prev_d = float(env._agent_distance_to_task(aid, prev_task))
                    step_now = int(env.state.step_index)
                    assigned = int(self.state.goal_assigned_step.get(str(aid), step_now))
                    hold_steps = int(self._goal_min_hold_steps(env, str(aid)))
                    within_hold = bool((step_now - assigned) < hold_steps)
                    near_enough = bool(np.isfinite(prev_d) and prev_d <= radius_m)
                    if within_hold or near_enough:
                        prev_keep_task = prev_task

                selected: List[object] = []
                if prev_keep_task is not None:
                    selected.append(prev_keep_task)
                for _, t in ranked:
                    if prev_keep_task is not None and str(t.task_id) == str(prev_keep_task.task_id):
                        continue
                    selected.append(t)
                    if len(selected) >= topk:
                        break
                if selected:
                    cands = selected

        for t in cands:
            tid = str(t.task_id)
            if tid in used:
                continue
            dist_norm = float(env._agent_distance_to_task(aid, t) / 3000.0)
            emer = 1.0 if str(t.kind.value) == "emergency" else 0.0
            tids.append(tid)
            feats.append([dist_norm, emer])
            task_nodes.append(int(t.demand_node))

        # Keep training/inference action space consistent:
        # add nearest-truck virtual candidate for UAV only when it is genuinely
        # needed for return/recovery, or when there is no currently available
        # emergency task candidate for this UAV.
        if st.kind == AgentKind.UAV and (not bool(st.crashed)):
            near_tid, near_dist = self._nearest_truck(env, aid)
            if near_tid is not None and np.isfinite(near_dist):
                truck_state = env.state.agents.get(str(near_tid), None)
                truck_node = int(getattr(truck_state, "node", 0) or 0)

                force_rth = bool((hasattr(env, "_uav_loaded") and (not bool(env._uav_loaded(aid)))) or float(getattr(st, "cargo", 0.0)) <= 0.0)
                if bool(getattr(env, "_uav_forced_rth_latch", {}).get(str(aid), False)):
                    force_rth = True
                if hasattr(env, "_uav_recovery_required"):
                    try:
                        if bool(env._uav_recovery_required(str(aid))):
                            force_rth = True
                    except Exception:
                        pass
                if not force_rth:
                    recovery_buf = float(max(getattr(env.cfg, "uav_recovery_distance_buffer_m", 600.0), 0.0))
                    req_batt = self._required_rth_battery(env, aid, float(near_dist) + recovery_buf)
                    safety_factor = float(max(getattr(env.cfg, "rth_safety_factor", 1.2), 0.0))
                    if float(getattr(st, "battery", 0.0)) < float(req_batt * safety_factor):
                        force_rth = True

                # If UAV is already docked, truck-recovery objective is already satisfied.
                # Keep emergency-task candidates visible to avoid truck<->task ABA bounce.
                if st.follow_target is not None:
                    force_rth = False

                allow_truck_candidate = bool(force_rth or len(tids) == 0)
                if allow_truck_candidate:
                    truck_feat = [float(near_dist / 3000.0), 0.0]
                    tids = [near_tid] + tids
                    feats = [truck_feat] + feats
                    task_nodes = [truck_node] + task_nodes
                if enable_mask and force_rth:
                    truck_feat = [float(near_dist / 3000.0), 0.0]
                    tids = [near_tid]
                    feats = [truck_feat]
                    task_nodes = [truck_node]

        return tids, feats, task_nodes

    def select_goals_with_policy(
        self,
        env,
        *,
        stochastic: bool,
        encoder_type: Optional[str] = None,
        enable_rth_mask: Optional[bool] = None,
        return_records: bool = False,
    ) -> Tuple[Dict[str, Optional[str]], List[Dict[str, object]]]:
        et = str(encoder_type or self.encoder_type).strip().lower()
        self._sync_risk_beta(env)
        used_tasks = set()
        goals: Dict[str, Optional[str]] = {}
        records: List[Dict[str, object]] = []
        graph_snapshot = self.capture_graph_snapshot(env) if et == "hetgat" else None
        high_dev = next(self.allocator.parameters()).device

        for aid in self._ordered_agents(env):
            st = env.state.agents[aid]
            if st.kind == AgentKind.UAV and bool(st.crashed):
                goals[aid] = None
                continue

            tids, feats, task_nodes = self.build_candidates_for_agent(
                env,
                aid,
                used_tasks=used_tasks,
                enable_rth_mask=enable_rth_mask,
            )
            if not tids:
                goals[aid] = None
                continue

            logit_bias = [0.0 for _ in tids]
            if st.kind == AgentKind.UAV:
                deadline_bias_enabled = bool(getattr(env.cfg, "uav_emergency_deadline_bias_enabled", True))
                deadline_slack_steps = int(max(getattr(env.cfg, "uav_emergency_deadline_bias_slack_steps", 24), 0))
                deadline_bias_weight = float(max(getattr(env.cfg, "uav_emergency_deadline_bias_weight", 0.30), 0.0))
                unassigned_bonus = float(max(getattr(env.cfg, "uav_emergency_unassigned_bonus", 0.18), 0.0))
                current_goal_ids = {
                    str(g)
                    for g in self.state.goals.values()
                    if g is not None
                }
                step_now = int(env.state.step_index)
                for i, tid in enumerate(tids):
                    t = env.state.tasks.get(str(tid), None)
                    if t is None:
                        continue
                    logit_bias[i] += self._uav_reachability_soft_bias(env, aid, t)
                    if (not deadline_bias_enabled) or t.kind != TaskKind.EMERGENCY or t.status != TaskStatus.PENDING:
                        continue
                    if deadline_slack_steps > 0:
                        slack = int(max(int(t.deadline_step) - step_now, 0))
                        urgency = float(np.clip((float(deadline_slack_steps) - float(slack)) / float(deadline_slack_steps), 0.0, 1.0))
                        logit_bias[i] += float(deadline_bias_weight * urgency)
                    if str(tid) not in current_goal_ids and str(tid) not in used_tasks:
                        logit_bias[i] += float(unassigned_bonus)
            elif st.kind == AgentKind.TRUCK:
                pending_norm = sum(
                    1
                    for t in env.state.tasks.values()
                    if t.status == TaskStatus.PENDING and t.kind == TaskKind.NORMAL
                )
                pending_emer = sum(
                    1
                    for t in env.state.tasks.values()
                    if t.status == TaskStatus.PENDING and t.kind == TaskKind.EMERGENCY
                )
                total_pending = float(max(pending_norm + pending_emer, 1))
                normal_pressure = float(np.clip(float(pending_norm) / total_pending, 0.0, 1.0))
                support_count = float(max(getattr(env, "truck_forward_support_count_total", 0), 0))
                truck_count = max(
                    1.0,
                    float(sum(1 for ag in env.state.agents.values() if ag.kind == AgentKind.TRUCK)),
                )
                support_sat = float(np.clip(support_count / (3.0 * truck_count), 0.0, 1.0))
                map_update_active = bool(self._last_refresh_flags.get("map_update", False))
                for i, tid in enumerate(tids):
                    t = env.state.tasks.get(str(tid), None)
                    if t is None:
                        continue
                    if t.kind == TaskKind.NORMAL:
                        logit_bias[i] += float(0.20 + 0.30 * normal_pressure + 0.08 * support_sat)
                    else:
                        # Under high NORMAL backlog, reduce truck emergency diversion unless
                        # emergency has explicit island/support context in lower layers.
                        logit_bias[i] -= float(0.16 * normal_pressure)
                    if map_update_active:
                        d = float(env._agent_distance_to_task(str(aid), t))
                        if np.isfinite(d):
                            map_prox = float(0.14 / (1.0 + d / 2500.0))
                            if t.kind == TaskKind.NORMAL:
                                logit_bias[i] += map_prox
                            else:
                                logit_bias[i] += 0.5 * map_prox

            # Soft goal inertia (no hard lock): slight preference to keep current goal
            # when it remains feasible/visible in current candidate set.
            prev_goal = self.state.goals.get(str(aid), None)
            prev_goal_idx = -1
            if prev_goal is not None:
                prev_goal = str(prev_goal)
                for i, tid in enumerate(tids):
                    if str(tid) == prev_goal:
                        prev_goal_idx = int(i)
                        keep_bonus = float(
                            max(getattr(env.cfg, "goal_keep_bias", 0.08), 0.0)
                        )
                        logit_bias[i] += keep_bonus
                        break

            base_x = torch.as_tensor(np.asarray(feats, dtype=np.float32), dtype=torch.float32, device=high_dev)
            enc = self.encode_candidates(
                base_feat=base_x,
                task_node_idx=task_nodes,
                encoder_type=et,
                graph_snapshot=graph_snapshot,
            )
            logits = self.allocator(enc).view(-1)
            if len(logit_bias) == int(logits.numel()):
                logits = logits + torch.as_tensor(logit_bias, dtype=logits.dtype, device=logits.device)
            if stochastic:
                sampled_idx = int(Categorical(logits=logits).sample().item())
            else:
                sampled_idx = int(torch.argmax(logits).item())

            chosen_tid = str(tids[sampled_idx])
            if st.kind == AgentKind.UAV:
                chosen_tid = self._resolve_uav_goal_with_lock(
                    env,
                    str(aid),
                    chosen_tid,
                    [str(x) for x in tids],
                    used_tasks,
                )
                if chosen_tid is None:
                    goals[aid] = None
                    continue

            chosen_tid = self._stabilize_goal_choice(
                env,
                str(aid),
                [str(x) for x in tids],
                logits,
                chosen_tid,
                used_tasks,
            )
            if chosen_tid is None:
                goals[aid] = None
                continue

            final_idx = sampled_idx
            for i, tid in enumerate(tids):
                if str(tid) == str(chosen_tid):
                    final_idx = int(i)
                    break
            chosen_tid = str(tids[final_idx])
            old_logp = float(torch.log_softmax(logits, dim=0)[final_idx].item())

            prev_goal_final = self.state.goals.get(str(aid), None)
            goals[aid] = chosen_tid
            if prev_goal_final != chosen_tid or str(aid) not in self.state.goal_assigned_step:
                self.state.goal_assigned_step[str(aid)] = int(env.state.step_index)

            chosen_task = env.state.tasks.get(str(chosen_tid), None)
            if chosen_task is not None and chosen_task.status == TaskStatus.PENDING:
                used_tasks.add(str(chosen_tid))
            if return_records:
                records.append(
                    {
                        "aid": aid,
                        "features": feats,
                        "task_nodes": list(task_nodes),
                        "graph_snapshot": graph_snapshot,
                        "candidate_ids": list(tids),
                        "logit_bias": list(logit_bias),
                        "prev_goal_idx": int(prev_goal_idx),
                        "action_idx": int(final_idx),
                        "old_logp": float(old_logp),
                        "return": 0.0,
                        "adv": 0.0,
                    }
                )

        # Truck anti-idle repair: ensure a reachable/serviceable task assignment
        # whenever pending work exists, reducing empty-goal drift.
        for aid in self._ordered_agents(env):
            st2 = env.state.agents.get(str(aid), None)
            if st2 is None or st2.kind != AgentKind.TRUCK or bool(getattr(st2, "crashed", False)):
                continue
            gid = goals.get(str(aid), None)
            task = env.state.tasks.get(str(gid), None) if gid is not None else None
            gid_ok = bool(
                task is not None
                and task.status == TaskStatus.PENDING
                and self._goal_valid_for_agent(env, str(aid), str(gid))
                and self._truck_task_reachable(env, str(aid), task)
            )
            if gid_ok:
                continue
            tid = self._nearest_truck_reachable_task(
                env,
                str(aid),
                used_tasks=used_tasks,
                allow_used=False,
            )
            if tid is None:
                tid = self._nearest_truck_reachable_task(
                    env,
                    str(aid),
                    used_tasks=used_tasks,
                    allow_used=True,
                )
            if tid is not None:
                goals[str(aid)] = str(tid)
                used_tasks.add(str(tid))

        return goals, records

    def _score_candidates(self, env, aid: str, tasks: List[object]) -> List[Tuple[str, float]]:
        if not tasks:
            return []
        feats = []
        tids = []
        task_nodes = []
        for t in tasks:
            dist_norm = float(env._agent_distance_to_task(aid, t) / 3000.0)
            emer = 1.0 if t.kind == TaskKind.EMERGENCY else 0.0
            feats.append([dist_norm, emer])
            tids.append(str(t.task_id))
            task_nodes.append(int(t.demand_node))
        base_x = torch.tensor(feats, dtype=torch.float32)
        graph = self.capture_graph_snapshot(env) if self.encoder_type == "hetgat" else None
        enc = self.encode_candidates(
            base_feat=base_x,
            task_node_idx=task_nodes,
            encoder_type=self.encoder_type,
            graph_snapshot=graph,
        )
        scores = self.allocator(enc).detach().cpu().tolist()
        return list(zip(tids, [float(v) for v in scores]))

    def _risk_node_embedding(self, env) -> Optional[torch.Tensor]:
        self._sync_risk_beta(env)
        n = len(env.topology.nodes)
        if n <= 0:
            return None
        x_rows = []
        for i in range(n):
            node = env.topology.nodes[i]
            hz = env.hazards.weather_at((float(node.x), float(node.y)))
            x_rows.append(
                [
                    float(hz.rain / max(env.cfg.base_rainfall_mmh, 1e-6)),
                    float(hz.wind / max(env.cfg.base_wind_mps, 1e-6)),
                    float(hz.quake),
                    float(node.slope_norm),
                ]
            )
        x = torch.tensor(x_rows, dtype=torch.float32)
        edges = []
        risks = []
        for src, nbs in env.topology.adjacency.items():
            for dst in nbs:
                if src == dst:
                    continue
                edges.append((int(src), int(dst)))
                k = (min(int(src), int(dst)), max(int(src), int(dst)))
                risks.append(float(env.hazards.last_edge_pstep.get(k, 0.0)))
        if not edges:
            return None
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_risk = torch.tensor(risks, dtype=torch.float32)
        emb, _ = self.risk_gat(x, edge_index, edge_risk=edge_risk)
        # Higher value = lower risk preference (inverse hazard energy proxy).
        return emb.norm(dim=-1)

    def plan(self, env) -> Dict[str, Optional[str]]:
        did_refresh = bool(self._should_refresh(env))
        if did_refresh:
            goals, _ = self.select_goals_with_policy(
                env,
                stochastic=False,
                encoder_type=self.encoder_type,
                enable_rth_mask=bool(getattr(env.cfg, "enable_rth_mask", True)),
                return_records=False,
            )
            self.state.goals = goals
            self.state.step_last_refresh = int(env.state.step_index)
            self.state.resolved_tasks_last = self._resolved_count(env)
            reasons = [k for k, v in self._last_refresh_flags.items() if k != "refresh" and bool(v)]
            self.last_replan_reason = "+".join(reasons) if reasons else "none"
        else:
            self.last_replan_reason = "no_refresh"

        if hasattr(env, "note_planner_replan"):
            try:
                env.note_planner_replan(dict(self._last_refresh_flags), str(self.last_replan_reason))
            except Exception:
                pass
        return dict(self.state.goals)





