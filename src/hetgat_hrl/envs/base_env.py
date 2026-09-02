from __future__ import annotations

import hashlib
import heapq
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
except Exception:  # pragma: no cover
    linear_sum_assignment = None

from hetgat_hrl.core.mdp_spec import (
    AgentKind,
    AgentRuntimeState,
    DeliveryTask,
    EnvConfig,
    HazardSnapshot,
    HeteroDisasterMDP,
    JointAction,
    JointState,
    StepResult,
    TaskKind,
    TaskClass,
    TaskStatus,
    TruckAction,
    UAVAction,
)
from hetgat_hrl.core.runtime_constants import DEPOT_DOCK_ID
from hetgat_hrl.core.algorithm_profile import (
    AlgorithmProfile,
    ER_HLNS_B_DOCKED_LATCH_REARM_CAPABILITY,
    ER_HLNS_B_PRELAUNCH_CONTRACT_LOCK_CAPABILITY,
    er_hlns_route_plan_active,
    uav_scout_information_active,
)
from hetgat_hrl.core.topology import GraphTopology
from hetgat_hrl.envs.base_env_runtime import init_base_env_runtime_state
from hetgat_hrl.envs.base_env_v2_runtime import (
    init_physical_v2_runtime,
    record_v2_energy_deduction,
    record_v2_recovery,
    record_v2_recovery_motion,
    record_v2_service_complete,
    record_v2_service_start,
    v2_authoritative_launch_check,
    v2_authorize_service_completion,
    v2_authorize_service_start,
    v2_edge_accessible,
    v2_metrics,
    v2_truck_speed_multiplier,
    v2_uav_energy_cost_fraction,
)
from hetgat_hrl.envs.base_env_step_validation import (
    make_invalid_action_record,
    normalize_truck_step_action,
    normalize_uav_step_action,
    safe_noop_for_agent_state,
    validate_action_for_dispatch,
)
from hetgat_hrl.envs.hazards import DynamicHazardField
from hetgat_hrl.envs.task_manager import DynamicTaskManager
from hetgat_hrl.hrl_v2.command_layer import CommandValidator


# Experimental only: the outbound-only delivery bypass is retained for
# controlled follow-up evaluation, but the paper/default runtime keeps the
# original forced-recovery motion gate until that path is revalidated.
_UAV_SORTIE_DELIVERY_LEG_RECOVERY_BYPASS_ENABLED = True

# Rejected by the seed117 A/B check: launching directly instead of completing
# an already planned truck-transfer saved one emergency task but displaced two
# bulk deliveries.  Keep the experiment available, disabled by default; the
# active repair below resumes the same task after the transfer bind instead.
_UAV_DIRECT_DELIVERY_OVER_TRANSFER_ENABLED = False
_UAV_DIRECT_DELIVERY_LONG_LEG_THRESHOLD_M = 4500.0
_UAV_NONPROGRESS_TRANSFER_FALLBACK_ENABLED = False
_UAV_TASK_TRANSFER_BIND_RELEASE_ENABLED = False
_UAV_POST_TRANSFER_CONTRACT_CONTINUATION_ENABLED = False
_UAV_TRANSFER_RECEIVER_PROGRESS_OVERRIDE_ENABLED = False


class BaseHeteroDisasterEnv(HeteroDisasterMDP):
    """
    Step-1 executable skeleton.
    This class only locks the MDP/SMDP interfaces and state transitions shape.
    """
    def __init__(
        self,
        cfg: Optional[EnvConfig] = None,
        algorithm_profile: Optional[AlgorithmProfile] = None,
    ):
        self.cfg = cfg or EnvConfig()
        self.algorithm_profile = algorithm_profile
        # Resolve legacy-vs-new naming into effective runtime values.
        self._num_nodes = int(self.cfg.num_nodes or self.cfg.n_nodes)
        self._num_trucks = int(self.cfg.num_trucks or self.cfg.n_trucks)
        self._num_uavs = int(self.cfg.num_uavs or self.cfg.n_uavs)
        _bulk_cfg = int(getattr(self.cfg, "num_routine_bulk_tasks", 0))
        _light_cfg = int(getattr(self.cfg, "num_time_critical_lightweight_tasks", 0))
        self._num_normal_tasks = int(_bulk_cfg if _bulk_cfg > 0 else (self.cfg.num_normal_tasks or self.cfg.n_normal_tasks))
        self._num_emergency_tasks = int(
            _light_cfg if _light_cfg > 0 else (self.cfg.num_emergency_tasks or self.cfg.n_emergency_tasks)
        )
        self._dt_seconds = float(self.cfg.dt_seconds)
        if self._dt_seconds <= 0.0:
            raise ValueError(f"dt_seconds must be > 0, got dt_seconds={self._dt_seconds}")

        # Seed-owned RNG must be initialized before any stochastic world construction.
        self.rng = np.random.default_rng(self.cfg.seed)
        self.topology = GraphTopology.build_from_config(self.cfg)
        self.hazards = DynamicHazardField(
            topo=self.topology,
            seed=self.cfg.seed + 1,
            stochastic_weather=self.cfg.stochastic_weather,
            cfg=self.cfg,
        )
        self.task_manager = DynamicTaskManager(self.topology)
        self.state = self._build_initial_state()
        self._init_comm_blackout_protocol()

        init_base_env_runtime_state(self)
        init_physical_v2_runtime(self)
        # These route-plan counters historically lived only in ``reset``.
        # Direct environment users (including checkpoint tests) must observe
        # the same initialized contract before the first step.
        self.routine_multiround_commitment_count = 0
        self.routine_multiround_support_block_count = 0
        self._routine_multiround_service_commitment_by_truck = {}

    def _task_candidate_node_ids(self, include_depot: bool = False) -> np.ndarray:
        node_ids = np.asarray(sorted(int(nid) for nid in self.topology.nodes.keys()), dtype=np.int64)
        if not bool(include_depot):
            node_ids = node_ids[node_ids != 0]
        if int(node_ids.size) <= 0:
            return np.asarray([0], dtype=np.int64)
        return node_ids

    def _sample_uniform_task_nodes(
        self,
        count: int,
        candidate_node_ids: Optional[List[int]] = None,
        replace: Optional[bool] = None,
    ) -> List[int]:
        if candidate_node_ids is None:
            node_ids = self._task_candidate_node_ids(include_depot=False)
        else:
            node_ids = np.asarray(sorted(int(n) for n in candidate_node_ids), dtype=np.int64)
        if int(count) <= 0 or int(node_ids.size) <= 0:
            return []
        rep = bool(int(count) > int(node_ids.size)) if replace is None else bool(replace)
        draw = self.rng.choice(node_ids, size=int(count), replace=rep)
        return [int(x) for x in np.atleast_1d(draw)]

    def _sample_emergency_task_nodes(
        self,
        count: int,
        candidate_node_ids: Optional[List[int]] = None,
        replace: Optional[bool] = None,
    ) -> List[int]:
        if candidate_node_ids is None:
            node_ids = self._task_candidate_node_ids(include_depot=False)
        else:
            node_ids = np.asarray(sorted(int(n) for n in candidate_node_ids), dtype=np.int64)
        if int(count) <= 0 or int(node_ids.size) <= 0:
            return []

        rep = bool(int(count) > int(node_ids.size)) if replace is None else bool(replace)
        bias = float(getattr(self.cfg, "task_emergency_epicenter_bias", 0.0))
        if bias <= 0.0:
            return self._sample_uniform_task_nodes(
                int(count),
                candidate_node_ids=[int(x) for x in node_ids],
                replace=rep,
            )

        # Disaster-demand coupling generation mechanism:
        # emergency demand concentrates around hazard epicenter while keeping a
        # uniform background component for reproducible long-tail coverage.
        epicenter_node_id = int(getattr(self.hazards, "epicenter_node", int(node_ids[0])))
        epi_node = self.topology.nodes.get(epicenter_node_id, self.topology.nodes[int(node_ids[0])])
        sigma_m = float(max(getattr(self.cfg, "task_emergency_sigma_m", 1200.0), 1.0))

        dists = np.asarray(
            [
                float(
                    np.hypot(
                        float(self.topology.nodes[int(nid)].x) - float(epi_node.x),
                        float(self.topology.nodes[int(nid)].y) - float(epi_node.y),
                    )
                )
                for nid in node_ids
            ],
            dtype=np.float64,
        )
        gauss = np.exp(-0.5 * np.square(dists / sigma_m))
        if (not np.isfinite(gauss).all()) or float(np.sum(gauss)) <= 0.0:
            return self._sample_uniform_task_nodes(
                int(count),
                candidate_node_ids=[int(x) for x in node_ids],
                replace=rep,
            )

        gauss = gauss / float(np.sum(gauss))
        uniform = np.full(shape=int(node_ids.size), fill_value=1.0 / float(node_ids.size), dtype=np.float64)
        alpha = float(np.clip(bias, 0.0, 1.0))
        probs = (1.0 - alpha) * uniform + alpha * gauss
        probs = probs / float(np.sum(probs))

        draw = self.rng.choice(node_ids, size=int(count), replace=rep, p=probs)
        return [int(x) for x in np.atleast_1d(draw)]


    def _real_case_task_sampling_enabled(self) -> bool:
        meta = getattr(self.topology, "real_case_meta", {})
        return bool(
            getattr(self.cfg, "real_case_enabled", False)
            and isinstance(meta, dict)
            and (meta.get("fixed_tasks") or meta.get("task_pools"))
        )

    def _synthetic_realism_task_sampling_enabled(self) -> bool:
        meta = getattr(self.topology, "real_case_meta", {})
        if not isinstance(meta, dict) or not bool(meta.get("synthetic_realism", False)):
            return False
        if not bool(getattr(self.cfg, "synthetic_realism_task_sampling_enabled", False)):
            return False
        min_size = float(getattr(self.cfg, "synthetic_realism_task_sampling_min_map_size_m", 10000.0))
        return bool(float(getattr(self.cfg, "map_size_m", 0.0)) >= min_size)

    def _node_dist_m(self, node_a: int, node_b: int) -> float:
        xa, ya = self._node_xy(int(node_a))
        xb, yb = self._node_xy(int(node_b))
        return float(np.hypot(float(xa) - float(xb), float(ya) - float(yb)))

    def _select_spaced_task_nodes(
        self,
        candidates: List[int],
        count: int,
        used_nodes: Optional[set] = None,
        min_spacing_m: Optional[float] = None,
        depot_exclusion_m: float = 500.0,
        preferred_cluster: Optional[int] = None,
    ) -> List[int]:
        if int(count) <= 0:
            return []
        used = used_nodes if used_nodes is not None else set()
        meta = getattr(self.topology, "real_case_meta", {}) if isinstance(getattr(self.topology, "real_case_meta", {}), dict) else {}
        cluster_map = dict(meta.get("cluster_id_by_node", {})) if isinstance(meta.get("cluster_id_by_node", {}), dict) else {}
        spacing = float(max(min_spacing_m if min_spacing_m is not None else getattr(self.cfg, "l_task_min_spacing_m", 240.0), 0.0))
        order = [int(n) for n in candidates if int(n) in self.topology.nodes and int(n) != 0]
        self.rng.shuffle(order)
        if preferred_cluster is not None:
            order.sort(key=lambda nid: 0 if int(cluster_map.get(int(nid), -1)) == int(preferred_cluster) else 1)
        selected: List[int] = []
        for nid in order:
            if int(nid) in used:
                continue
            if float(self._node_dist_m(0, int(nid))) < float(depot_exclusion_m):
                continue
            too_close = False
            for other in list(used) + selected:
                if float(self._node_dist_m(int(other), int(nid))) < spacing:
                    too_close = True
                    break
            if too_close:
                continue
            selected.append(int(nid))
            used.add(int(nid))
            if len(selected) >= int(count):
                break
        return selected

    def _sample_real_case_task_nodes(self, normal_count: int, emergency_count: int) -> Tuple[List[int], List[int]]:
        meta = getattr(self.topology, "real_case_meta", {}) if isinstance(getattr(self.topology, "real_case_meta", {}), dict) else {}
        fixed = list(meta.get("fixed_tasks", [])) if isinstance(meta.get("fixed_tasks", []), list) else []
        if fixed:
            normal_nodes = [
                int(item["node_id"])
                for item in fixed
                if str(item.get("task_class", "")).strip().lower() in {"routine_bulk", "basic", "normal"}
            ]
            emergency_nodes = [
                int(item["node_id"])
                for item in fixed
                if str(item.get("task_class", "")).strip().lower()
                in {"time_critical_lightweight", "emergency"}
            ]
            if len(normal_nodes) != int(normal_count) or len(emergency_nodes) != int(emergency_count):
                raise ValueError(
                    "Fixed RB task manifest count mismatch: "
                    f"expected {normal_count}+{emergency_count}, got "
                    f"{len(normal_nodes)}+{len(emergency_nodes)}"
                )
            all_nodes = normal_nodes + emergency_nodes
            if 0 in all_nodes or len(set(all_nodes)) != len(all_nodes):
                raise ValueError("Fixed RB task nodes must be unique and cannot equal the remapped depot node 0")
            return normal_nodes, emergency_nodes
        pools = dict(meta.get("task_pools", {})) if isinstance(meta.get("task_pools", {}), dict) else {}
        clusters = list(meta.get("major_clusters", [])) if isinstance(meta.get("major_clusters", []), list) else []
        cluster_map = dict(meta.get("cluster_id_by_node", {})) if isinstance(meta.get("cluster_id_by_node", {}), dict) else {}
        used: set = set()
        normal_nodes: List[int] = []
        emergency_nodes: List[int] = []

        total_tasks = int(normal_count) + int(emergency_count)
        if total_tasks >= 12 and clusters:
            for cid, _ in enumerate(clusters):
                if len(normal_nodes) < int(normal_count):
                    cluster_bulk = [
                        int(n) for n in pools.get("bulk_builtup", [])
                        if int(cluster_map.get(int(n), -1)) == int(cid)
                    ]
                    normal_nodes.extend(self._select_spaced_task_nodes(cluster_bulk, 1, used_nodes=used, preferred_cluster=cid))
                if len(emergency_nodes) < int(emergency_count):
                    cluster_tc = [
                        int(n) for n in (
                            list(pools.get("timecritical_gateway", []))
                            + list(pools.get("timecritical_medical", []))
                            + list(pools.get("timecritical_hazard", []))
                        )
                        if int(cluster_map.get(int(n), -1)) == int(cid)
                    ]
                    emergency_nodes.extend(self._select_spaced_task_nodes(cluster_tc, 1, used_nodes=used, preferred_cluster=cid))

        bulk_plan = [
            ("bulk_builtup", 0.55),
            ("bulk_gateway", 0.25),
            ("bulk_peripheral", 0.20),
        ]
        tc_plan = [
            ("timecritical_medical", 0.40),
            ("timecritical_gateway", 0.35),
            ("timecritical_hazard", 0.25),
        ]
        normal_remaining = max(int(normal_count) - len(normal_nodes), 0)
        emergency_remaining = max(int(emergency_count) - len(emergency_nodes), 0)

        for idx, (pool_name, frac) in enumerate(bulk_plan):
            if normal_remaining <= 0:
                break
            want = normal_remaining if idx == len(bulk_plan) - 1 else int(round(float(normal_count) * frac))
            want = max(0, min(want, normal_remaining))
            picked = self._select_spaced_task_nodes(list(pools.get(pool_name, [])), want, used_nodes=used)
            normal_nodes.extend(picked)
            normal_remaining = max(int(normal_count) - len(normal_nodes), 0)

        for idx, (pool_name, frac) in enumerate(tc_plan):
            if emergency_remaining <= 0:
                break
            want = emergency_remaining if idx == len(tc_plan) - 1 else int(round(float(emergency_count) * frac))
            want = max(0, min(want, emergency_remaining))
            picked = self._select_spaced_task_nodes(list(pools.get(pool_name, [])), want, used_nodes=used)
            emergency_nodes.extend(picked)
            emergency_remaining = max(int(emergency_count) - len(emergency_nodes), 0)

        if len(normal_nodes) < int(normal_count):
            bulk_fallback = list(dict.fromkeys(
                list(pools.get("bulk_builtup", []))
                + list(pools.get("bulk_gateway", []))
                + list(pools.get("bulk_peripheral", []))
            ))
            normal_nodes.extend(self._select_spaced_task_nodes(bulk_fallback, int(normal_count) - len(normal_nodes), used_nodes=used, min_spacing_m=0.75 * float(getattr(self.cfg, "l_task_min_spacing_m", 240.0))))
        if len(emergency_nodes) < int(emergency_count):
            tc_fallback = list(dict.fromkeys(
                list(pools.get("timecritical_medical", []))
                + list(pools.get("timecritical_gateway", []))
                + list(pools.get("timecritical_hazard", []))
                + list(pools.get("bulk_gateway", []))
            ))
            emergency_nodes.extend(self._select_spaced_task_nodes(tc_fallback, int(emergency_count) - len(emergency_nodes), used_nodes=used, min_spacing_m=0.65 * float(getattr(self.cfg, "l_task_min_spacing_m", 240.0))))

        return [int(x) for x in normal_nodes[: int(normal_count)]], [int(x) for x in emergency_nodes[: int(emergency_count)]]

    def _sample_synthetic_realism_task_nodes(self, normal_count: int, emergency_count: int) -> Tuple[List[int], List[int]]:
        meta = getattr(self.topology, "real_case_meta", {}) if isinstance(getattr(self.topology, "real_case_meta", {}), dict) else {}
        tasks = dict(meta.get("synthetic_realism_tasks", {})) if isinstance(meta.get("synthetic_realism_tasks", {}), dict) else {}

        def nodes_from(items: Any) -> List[int]:
            out: List[int] = []
            if not isinstance(items, list):
                return out
            for item in items:
                if not isinstance(item, dict):
                    continue
                nid = int(item.get("node_id", -1))
                if nid in self.topology.nodes and nid != 0 and nid not in out:
                    out.append(int(nid))
            return out

        normal_nodes = nodes_from(tasks.get("normal", []))
        emergency_nodes = nodes_from(tasks.get("emergency", []))
        used = set(int(x) for x in normal_nodes + emergency_nodes)

        if len(normal_nodes) < int(normal_count) or len(emergency_nodes) < int(emergency_count):
            fallback = [
                int(nid)
                for nid in self._task_candidate_node_ids(include_depot=False).tolist()
                if int(nid) not in used
            ]
            self.rng.shuffle(fallback)
            if len(normal_nodes) < int(normal_count):
                take = fallback[: max(int(normal_count) - len(normal_nodes), 0)]
                normal_nodes.extend(int(x) for x in take)
                used.update(int(x) for x in take)
                fallback = [int(x) for x in fallback if int(x) not in used]
            if len(emergency_nodes) < int(emergency_count):
                take = fallback[: max(int(emergency_count) - len(emergency_nodes), 0)]
                emergency_nodes.extend(int(x) for x in take)

        return [int(x) for x in normal_nodes[: int(normal_count)]], [int(x) for x in emergency_nodes[: int(emergency_count)]]

    def _initial_route_dispatch_outlet_nodes(self) -> List[int]:
        if not bool(getattr(self.cfg, "hrl_initial_route_dispatch_enabled", True)):
            return []
        try:
            nodes = list(self._decision_neighbors(0))
        except Exception:
            try:
                nodes = list(self.topology.neighbors(0))
            except Exception:
                nodes = []
        return sorted(int(n) for n in nodes if int(n) != 0)

    def _initial_route_task_bucket(self, task: DeliveryTask, outlets: Optional[List[int]] = None) -> Optional[int]:
        if task is None:
            return None
        outlet_nodes = list(outlets) if outlets is not None else self._initial_route_dispatch_outlet_nodes()
        if len(outlet_nodes) < 2:
            return None
        task_node = int(task.demand_node)
        best_idx: Optional[int] = None
        best_key: Tuple[float, float, int] = (float("inf"), float("inf"), 10**9)
        tx, ty = self._node_xy(int(task_node))
        for idx, outlet in enumerate(outlet_nodes):
            try:
                sp = float(self._decision_shortest_path_distance(int(outlet), int(task_node)))
            except Exception:
                sp = float("inf")
            if not np.isfinite(sp):
                continue
            ox, oy = self._node_xy(int(outlet))
            dxy = float(np.hypot(float(tx) - float(ox), float(ty) - float(oy)))
            key = (float(sp), float(dxy), int(idx))
            if key < best_key:
                best_key = key
                best_idx = int(idx)
        return best_idx

    def _initial_route_docked_uav_assignment(
        self,
        tasks: Dict[str, DeliveryTask],
        truck_ids: List[str],
        uav_ids: List[str],
        follower_cap: int,
    ) -> Dict[str, str]:
        if (
            not bool(getattr(self.cfg, "hrl_initial_route_dispatch_enabled", True))
            or not bool(getattr(self.cfg, "hrl_initial_route_docked_assignment_enabled", True))
            or len(truck_ids) <= 0
            or len(uav_ids) <= 0
            or int(follower_cap) <= 0
        ):
            return {}

        outlets = self._initial_route_dispatch_outlet_nodes()
        if len(outlets) < 2:
            return {}

        bucket_count = int(max(len(outlets), 1))
        stats: Dict[int, Dict[str, float]] = {
            int(i): {"normal": 0.0, "emergency": 0.0, "weighted": 0.0}
            for i in range(bucket_count)
        }
        for task in tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            bucket = self._initial_route_task_bucket(task, outlets=outlets)
            if bucket is None or int(bucket) < 0:
                continue
            if task.kind == TaskKind.EMERGENCY:
                stats[int(bucket)]["emergency"] += 1.0
                stats[int(bucket)]["weighted"] += 2.2 + 0.35 * float(np.clip(getattr(task, "urgency_score", 0.0), 0.0, 1.0))
            else:
                stats[int(bucket)]["normal"] += 1.0
                stats[int(bucket)]["weighted"] += 1.0 + 0.20 * float(np.clip(getattr(task, "urgency_score", 0.0), 0.0, 1.0))

        ranked_buckets = sorted(
            [int(i) for i in range(bucket_count)],
            key=lambda i: (
                float(stats[int(i)]["weighted"]),
                float(stats[int(i)]["emergency"]),
                -int(i),
            ),
            reverse=True,
        )
        if not ranked_buckets:
            return {}

        truck_bucket: Dict[str, int] = {}
        bucket_use_count: Dict[int, int] = {int(i): 0 for i in range(bucket_count)}
        nonempty = [int(i) for i in ranked_buckets if float(stats[int(i)]["weighted"]) > 1e-9]
        for idx, tid in enumerate(truck_ids):
            preferred = int(nonempty[idx]) if idx < len(nonempty) else None
            best_bucket = int(preferred if preferred is not None else ranked_buckets[0])
            best_score = -1e18
            candidates = list(nonempty) if nonempty else list(ranked_buckets)
            for bucket in candidates:
                weighted = float(stats[int(bucket)]["weighted"])
                if weighted <= 0.0:
                    continue
                score = float(
                    weighted
                    + 0.40 * float(stats[int(bucket)]["emergency"])
                    - 0.60 * float(bucket_use_count.get(int(bucket), 0))
                )
                if preferred is not None and int(bucket) == int(preferred):
                    score += 0.25 * max(weighted, 1.0)
                if score > best_score + 1e-12:
                    best_score = float(score)
                    best_bucket = int(bucket)
            truck_bucket[str(tid)] = int(best_bucket)
            bucket_use_count[int(best_bucket)] = int(bucket_use_count.get(int(best_bucket), 0) + 1)

        bucket_to_trucks: Dict[int, List[str]] = {int(i): [] for i in range(bucket_count)}
        for tid in truck_ids:
            bucket_to_trucks[int(truck_bucket.get(str(tid), 0))].append(str(tid))

        emergency_total = float(sum(float(v["emergency"]) for v in stats.values()))
        quota_by_bucket: Dict[int, int] = {int(i): 0 for i in range(bucket_count)}
        uav_total = int(len(uav_ids))
        if emergency_total > 1e-9:
            raw_share = {
                int(i): float(stats[int(i)]["emergency"]) / emergency_total
                for i in range(bucket_count)
            }
            for i in range(bucket_count):
                if float(raw_share[int(i)]) > 1e-9:
                    quota_by_bucket[int(i)] = int(max(1, int(np.floor(float(uav_total) * float(raw_share[int(i)])))))
            while int(sum(quota_by_bucket.values())) > int(uav_total):
                k = max(quota_by_bucket.keys(), key=lambda kk: quota_by_bucket[int(kk)])
                if quota_by_bucket[int(k)] <= 0:
                    break
                quota_by_bucket[int(k)] -= 1
            while int(sum(quota_by_bucket.values())) < int(uav_total):
                k = max(
                    range(bucket_count),
                    key=lambda kk: float(raw_share[int(kk)]) - float(quota_by_bucket[int(kk)]) / max(float(uav_total), 1.0),
                )
                quota_by_bucket[int(k)] += 1
        else:
            for idx, _uid in enumerate(uav_ids):
                quota_by_bucket[int(idx % bucket_count)] += 1

        assigned_by_truck: Dict[str, int] = {str(tid): 0 for tid in truck_ids}
        assigned_by_bucket: Dict[int, int] = {int(i): 0 for i in range(bucket_count)}
        assign_map: Dict[str, str] = {}
        for uid in uav_ids:
            unmet = [int(i) for i in range(bucket_count) if assigned_by_bucket[int(i)] < int(quota_by_bucket.get(int(i), 0))]
            if unmet:
                choose_bucket = max(unmet, key=lambda i: (float(stats[int(i)]["emergency"]), float(stats[int(i)]["weighted"]), -int(i)))
            else:
                choose_bucket = ranked_buckets[0]
            trucks_in_bucket = list(bucket_to_trucks.get(int(choose_bucket), []))
            if not trucks_in_bucket:
                trucks_in_bucket = list(truck_ids)
            available = [
                str(tid)
                for tid in trucks_in_bucket
                if int(assigned_by_truck.get(str(tid), 0)) < int(follower_cap)
            ]
            if available:
                trucks_in_bucket = available
            else:
                global_available = [
                    str(tid)
                    for tid in truck_ids
                    if int(assigned_by_truck.get(str(tid), 0)) < int(follower_cap)
                ]
                if global_available:
                    trucks_in_bucket = global_available
            if not trucks_in_bucket:
                continue
            best_tid = min(
                trucks_in_bucket,
                key=lambda tid: (
                    int(assigned_by_truck.get(str(tid), 0)),
                    abs(int(truck_bucket.get(str(tid), 0)) - int(choose_bucket)),
                    str(tid),
                ),
            )
            assign_map[str(uid)] = str(best_tid)
            assigned_by_truck[str(best_tid)] = int(assigned_by_truck.get(str(best_tid), 0) + 1)
            assigned_by_bucket[int(choose_bucket)] = int(assigned_by_bucket.get(int(choose_bucket), 0) + 1)
        return assign_map

    def _apply_forced_island_task_blocking(self, tasks: Dict[str, DeliveryTask]) -> None:
        self._forced_island_edge_keys = set()
        self._forced_island_task_ids = set()
        self._forced_island_candidate_task_ids = set()

        forced_count = int(max(getattr(self.cfg, "forced_island_emergency_tasks", 0), 0))
        if forced_count <= 0:
            return

        depot_xy = self._node_xy(0)
        candidates: List[Tuple[float, float, str, int]] = []
        for task in tasks.values():
            if task.kind != TaskKind.EMERGENCY:
                continue
            node = int(task.demand_node)
            if node == 0:
                continue
            nbs = list(self.topology.adjacency.get(node, set()))
            if not nbs:
                continue
            nxy = self._node_xy(node)
            dist_to_depot = float(np.hypot(float(nxy[0]) - float(depot_xy[0]), float(nxy[1]) - float(depot_xy[1])))
            degree = float(len(nbs))
            impact_score = float(dist_to_depot / max(degree, 1.0))
            candidates.append((impact_score, degree, str(task.task_id), int(node)))

        if not candidates:
            return

        # Prefer far and low-degree emergency nodes, so forced-island setup
        # does not inflate initial blocked ratio too much.
        candidates.sort(key=lambda x: (float(x[0]), -float(x[1])), reverse=True)
        selected: List[Tuple[str, int]] = []
        used_nodes = set()
        for _, _, tid, node in candidates:
            if int(node) in used_nodes:
                continue
            selected.append((str(tid), int(node)))
            used_nodes.add(int(node))
            if len(selected) >= forced_count:
                break

        # Keep the same candidate tasks and deadline contract in paired A/B/C
        # instances. Scenario A is the no-disaster control, so its roads remain
        # fully open; B/C apply the identical forced-island realization.
        self._forced_island_candidate_task_ids = {
            str(tid) for tid, _node in selected
        }
        if str(getattr(self.cfg, "scenario", "B")).upper().strip() == "A":
            return

        for tid, node in selected:
            for nb in list(self.topology.adjacency.get(int(node), set())):
                self.topology.set_blocked(int(node), int(nb), True)
                ek = (min(int(node), int(nb)), max(int(node), int(nb)))
                self._forced_island_edge_keys.add(ek)
            self._forced_island_task_ids.add(str(tid))

    def _enforce_forced_island_blocking(self) -> None:
        if not bool(getattr(self.cfg, "forced_island_lock_edges", True)):
            return
        if not self._forced_island_edge_keys:
            return
        for ea, eb in list(self._forced_island_edge_keys):
            self.topology.set_blocked(int(ea), int(eb), True)

    def _build_initial_state(self) -> JointState:
        agents: Dict[str, AgentRuntimeState] = {}
        truck_ids = [f"truck_{i}" for i in range(self._num_trucks)]
        start_docked = bool(getattr(self.cfg, "uav_start_docked_on_truck", False))
        follower_cap = int(max(getattr(self.cfg, "uav_max_followers_per_truck", 0), 0))
        uav_ids = [f"uav_{i}" for i in range(self._num_uavs)]

        # Allocate every initially loaded UAV to a truck-side emergency quota
        # before constructing truck inventory.  This avoids the historical
        # double count in which each truck received its complete emergency
        # stock and every UAV then received an additional package for free.
        initial_uav_home_by_id: Dict[str, Optional[str]] = {}
        home_counts: Dict[str, int] = {str(tid): 0 for tid in truck_ids}
        for i, uav_id in enumerate(uav_ids):
            home_tid: Optional[str] = None
            if truck_ids and follower_cap > 0:
                candidate = str(truck_ids[i % len(truck_ids)])
                if int(home_counts.get(candidate, 0)) < follower_cap:
                    home_tid = candidate
                else:
                    for tid in truck_ids:
                        tid_s = str(tid)
                        if int(home_counts.get(tid_s, 0)) < follower_cap:
                            home_tid = tid_s
                            break
            if home_tid is not None:
                home_counts[home_tid] = int(home_counts.get(home_tid, 0)) + 1
            initial_uav_home_by_id[str(uav_id)] = home_tid

        self._initial_uav_home_truck_by_uav = dict(initial_uav_home_by_id)
        tc_package_kg = float(self._timecritical_supply_unit_kg())
        initial_tc_quota_kg = float(self._truck_initial_timecritical_inventory_kg())
        for i in range(self._num_trucks):
            truck_id = f"truck_{i}"
            init_bulk_kg = float(self._truck_initial_bulk_inventory_kg())
            preloaded_uav_kg = float(home_counts.get(truck_id, 0)) * tc_package_kg
            if preloaded_uav_kg > initial_tc_quota_kg + 1e-9:
                raise ValueError(
                    "initial UAV emergency preload exceeds host truck quota: "
                    f"{truck_id} has {preloaded_uav_kg:.3f} kg > "
                    f"{initial_tc_quota_kg:.3f} kg"
                )
            init_tc_kg = float(max(initial_tc_quota_kg - preloaded_uav_kg, 0.0))
            full_inv_kg = float(init_bulk_kg + init_tc_kg)
            bulk_units = int(np.floor(init_bulk_kg / max(self._bulk_supply_unit_kg(), 1e-6)))
            tc_units = int(np.floor(init_tc_kg / max(self._timecritical_supply_unit_kg(), 1e-6)))
            agents[truck_id] = AgentRuntimeState(
                agent_id=truck_id,
                kind=AgentKind.TRUCK,
                node=0,
                pos_xy=self._node_xy(0),
                battery=1.0,
                cargo=float(full_inv_kg / max(float(self.cfg.cargo_unit_kg), 1e-6)),
                replenish_timer=0,
                normal_supply_units=int(bulk_units),
                emergency_supply_units=int(tc_units),
                bulk_inventory_kg_current=float(init_bulk_kg),
                timecritical_inventory_kg_current=float(init_tc_kg),
                truck_inventory_kg_current=float(full_inv_kg),
                truck_needs_replenish_flag=False,
                truck_replenish_timer=0,
            )
        for i in range(self._num_uavs):
            home_tid = initial_uav_home_by_id.get(f"uav_{i}")
            dock_tid = str(home_tid) if start_docked and home_tid is not None else None
            agents[f"uav_{i}"] = AgentRuntimeState(
                agent_id=f"uav_{i}",
                kind=AgentKind.UAV,
                node=0,
                pos_xy=self._node_xy(0),
                vel_xy=(0.0, 0.0),
                battery=float(self.cfg.uav_battery_init),
                cargo=float(self.cfg.uav_cargo_capacity_units),
                replenish_timer=0,
                follow_target=dock_tid,
                carried_emergency_units=1,
                payload_kg_current=float(self._timecritical_supply_unit_kg()),
                uav_needs_reload_flag=False,
                uav_reload_timer=0,
            )

        tasks: Dict[str, DeliveryTask] = {}
        # Task node sampling is seed-controlled and excludes depot by default.
        # Real-city cases use role-aware sampling pools (built-up centers, gateways,
        # bridgeheads, hazard-adjacent edges); synthetic maps keep the legacy seeded
        # uniform / epicenter-biased behavior.
        if self._real_case_task_sampling_enabled():
            normal_nodes, emergency_nodes = self._sample_real_case_task_nodes(
                int(self._num_normal_tasks),
                int(self._num_emergency_tasks),
            )
        elif self._synthetic_realism_task_sampling_enabled():
            normal_nodes, emergency_nodes = self._sample_synthetic_realism_task_nodes(
                int(self._num_normal_tasks),
                int(self._num_emergency_tasks),
            )
        else:
            candidate_nodes = [
                int(nid)
                for nid in self._task_candidate_node_ids(include_depot=False).tolist()
            ]
            normal_take = int(min(self._num_normal_tasks, len(candidate_nodes)))
            normal_nodes = self._sample_uniform_task_nodes(
                normal_take,
                candidate_node_ids=candidate_nodes,
                replace=False,
            )
            normal_node_set = set(int(x) for x in normal_nodes)
            remain_nodes = [int(n) for n in candidate_nodes if int(n) not in normal_node_set]
            emer_take_unique = int(min(self._num_emergency_tasks, len(remain_nodes)))
            emergency_nodes = self._sample_emergency_task_nodes(
                emer_take_unique,
                candidate_node_ids=remain_nodes,
                replace=False,
            )
            if int(self._num_emergency_tasks) > int(emer_take_unique):
                emergency_nodes.extend(
                    self._sample_emergency_task_nodes(
                        int(self._num_emergency_tasks) - int(emer_take_unique),
                        candidate_node_ids=candidate_nodes,
                        replace=True,
                    )
                )
            if int(self._num_normal_tasks) > int(normal_take):
                normal_nodes.extend(
                    self._sample_uniform_task_nodes(
                        int(self._num_normal_tasks) - int(normal_take),
                        candidate_node_ids=candidate_nodes,
                        replace=True,
                    )
                )

        unit_kg = float(max(float(getattr(self.cfg, "cargo_unit_kg", 200.0)), 1e-6))
        real_meta = getattr(self.topology, "real_case_meta", {})
        fixed_items = list(real_meta.get("fixed_tasks", [])) if isinstance(real_meta, dict) else []
        fixed_normal = [
            item for item in fixed_items
            if str(item.get("task_class", "")).strip().lower() in {"routine_bulk", "basic", "normal"}
        ]
        fixed_emergency = [
            item for item in fixed_items
            if str(item.get("task_class", "")).strip().lower()
            in {"time_critical_lightweight", "emergency"}
        ]
        for i, demand in enumerate(normal_nodes):
            fixed_item = fixed_normal[i] if i < len(fixed_normal) else None
            demand_kg = float(fixed_item.get("demand_kg", self.cfg.normal_task_demand_kg)) if fixed_item else float(
                self.rng.uniform(float(getattr(self.cfg, "routine_bulk_demand_kg_min", self.cfg.normal_task_demand_kg)), float(getattr(self.cfg, "routine_bulk_demand_kg_max", self.cfg.normal_task_demand_kg)))
            )
            urg = float(fixed_item.get("urgency_score", 0.40)) if fixed_item else float(
                self.rng.uniform(float(getattr(self.cfg, "routine_bulk_urgency_min", 0.20)), float(getattr(self.cfg, "routine_bulk_urgency_max", 0.60)))
            )
            decay = float(getattr(self.cfg, "routine_bulk_lifeline_decay_base", 0.08)) * float(0.6 + urg)
            demand_units = int(max(1, int(np.ceil(demand_kg / unit_kg))))
            tid = str(fixed_item.get("task_id", f"task_routine_bulk_{i}")) if fixed_item else f"task_routine_bulk_{i}"
            tasks[tid] = DeliveryTask(
                task_id=tid,
                kind=TaskKind.NORMAL,
                task_class=TaskClass.ROUTINE_BULK.value,
                demand_node=int(demand),
                deadline_step=int(fixed_item.get("deadline_step")) if fixed_item and "deadline_step" in fixed_item else min(self.cfg.max_steps - 1, int(max(getattr(self.cfg, "normal_task_deadline_start_step", 120), 0)) + i * int(max(getattr(self.cfg, "normal_task_deadline_interval_step", 5), 0))),
                status=TaskStatus.PENDING,
                demand_left=float(demand_units),
                remaining_demand_kg=float(demand_kg),
                demand_kg=float(demand_kg),
                demand_units=int(demand_units),
                supply_units_required=1,
                supply_type="normal",
                urgency_score=float(np.clip(urg, 0.0, 1.0)),
                lifeline_init=float(getattr(self.cfg, "task_lifeline_init_default", 100.0)),
                lifeline_current=float(getattr(self.cfg, "task_lifeline_init_default", 100.0)),
                lifeline_decay_rate=float(max(decay, 0.0)),
                created_step=0,
            )
        for i, demand in enumerate(emergency_nodes):
            fixed_item = fixed_emergency[i] if i < len(fixed_emergency) else None
            demand_kg = float(fixed_item.get("demand_kg", self.cfg.emergency_task_demand_kg)) if fixed_item else float(
                self.rng.uniform(float(getattr(self.cfg, "time_critical_lightweight_demand_kg_min", self.cfg.emergency_task_demand_kg)), float(getattr(self.cfg, "time_critical_lightweight_demand_kg_max", self.cfg.emergency_task_demand_kg)))
            )
            urg = float(fixed_item.get("urgency_score", 0.85)) if fixed_item else float(
                self.rng.uniform(float(getattr(self.cfg, "time_critical_lightweight_urgency_min", 0.70)), float(getattr(self.cfg, "time_critical_lightweight_urgency_max", 1.00)))
            )
            decay = float(getattr(self.cfg, "time_critical_lightweight_lifeline_decay_base", 0.22)) * float(0.6 + urg)
            demand_units = int(max(1, int(np.ceil(demand_kg / unit_kg))))
            tid = str(fixed_item.get("task_id", f"task_time_critical_lightweight_{i}")) if fixed_item else f"task_time_critical_lightweight_{i}"
            tasks[tid] = DeliveryTask(
                task_id=tid,
                kind=TaskKind.EMERGENCY,
                task_class=TaskClass.TIME_CRITICAL_LIGHTWEIGHT.value,
                demand_node=int(demand),
                deadline_step=int(fixed_item.get("deadline_step")) if fixed_item and "deadline_step" in fixed_item else min(self.cfg.max_steps - 1, int(max(getattr(self.cfg, "emergency_task_deadline_start_step", 80), 0)) + i * int(max(getattr(self.cfg, "emergency_task_deadline_interval_step", 4), 0))),
                status=TaskStatus.PENDING,
                demand_left=float(demand_units),
                remaining_demand_kg=float(demand_kg),
                demand_kg=float(demand_kg),
                demand_units=int(demand_units),
                supply_units_required=1,
                supply_type="emergency",
                urgency_score=float(np.clip(urg, 0.0, 1.0)),
                lifeline_init=float(getattr(self.cfg, "task_lifeline_init_default", 100.0)),
                lifeline_current=float(getattr(self.cfg, "task_lifeline_init_default", 100.0)),
                lifeline_decay_rate=float(max(decay, 0.0)),
                created_step=0,
            )

        self._apply_forced_island_task_blocking(tasks)
        # Forced-island emergency tasks are intentionally UAV-only under blocked roads;
        # give them controlled extra deadline slack so they are not systemically
        # doomed by construction under safety/recovery constraints.
        island_deadline_extra = int(max(getattr(self.cfg, "forced_island_deadline_extension_steps", 24), 0))
        if island_deadline_extra > 0:
            for tid in sorted(self._forced_island_candidate_task_ids):
                t = tasks.get(str(tid), None)
                if t is None or t.kind != TaskKind.EMERGENCY:
                    continue
                t.deadline_step = int(min(self.cfg.max_steps - 1, int(t.deadline_step) + island_deadline_extra))

        self._initial_route_docked_truck_by_uav = {}
        self.initial_route_docked_uav_count = 0
        if start_docked and truck_ids and uav_ids and follower_cap > 0:
            dock_assign = self._initial_route_docked_uav_assignment(
                tasks=tasks,
                truck_ids=[str(tid) for tid in truck_ids],
                uav_ids=[str(uid) for uid in uav_ids],
                follower_cap=int(follower_cap),
            )
            if dock_assign:
                for uid in uav_ids:
                    st_uav = agents.get(str(uid), None)
                    if st_uav is None or st_uav.kind != AgentKind.UAV:
                        continue
                    follow_tid = dock_assign.get(str(uid), None)
                    if follow_tid is None:
                        continue
                    st_uav.follow_target = str(follow_tid)
                self._initial_route_docked_truck_by_uav = dict(dock_assign)
                self.initial_route_docked_uav_count = int(len(dock_assign))
            else:
                self._initial_route_docked_truck_by_uav = {
                    str(uid): str(getattr(agents[str(uid)], "follow_target", ""))
                    for uid in uav_ids
                    if agents.get(str(uid), None) is not None and getattr(agents[str(uid)], "follow_target", None) is not None
                }
                self.initial_route_docked_uav_count = int(len(self._initial_route_docked_truck_by_uav))

        # The route-aware docking assignment above may differ from the
        # preliminary round-robin homes used while constructing agents. Make
        # the final assignment authoritative for material accounting so every
        # truck--UAV unit owns exactly four emergency packages for every seed.
        if start_docked:
            final_home_by_uav = {
                str(uid): str(getattr(agents[str(uid)], "follow_target", ""))
                for uid in uav_ids
                if agents.get(str(uid), None) is not None
                and getattr(agents[str(uid)], "follow_target", None) is not None
            }
            self._initial_uav_home_truck_by_uav = dict(final_home_by_uav)
            for truck_id in truck_ids:
                truck = agents[str(truck_id)]
                final_bulk_kg = float(self._truck_initial_bulk_inventory_kg())
                mounted_tc_kg = float(
                    sum(
                        max(float(getattr(agents[str(uid)], "payload_kg_current", 0.0)), 0.0)
                        for uid in uav_ids
                        if agents.get(str(uid), None) is not None
                        and str(getattr(agents[str(uid)], "follow_target", "")) == str(truck_id)
                        and str(getattr(agents[str(uid)], "payload_supply_type", "emergency")).strip().lower()
                        == "emergency"
                    )
                )
                final_tc_kg = float(max(initial_tc_quota_kg - mounted_tc_kg, 0.0))
                final_body_kg = float(final_bulk_kg + final_tc_kg)
                if final_body_kg + mounted_tc_kg > float(self.cfg.truck_payload_capacity_kg) + 1e-9:
                    raise ValueError(
                        "final route-aware UAV preload exceeds truck material capacity: "
                        f"{truck_id} has {final_body_kg + mounted_tc_kg:.3f} kg > "
                        f"{float(self.cfg.truck_payload_capacity_kg):.3f} kg"
                    )
                truck.bulk_inventory_kg_current = float(final_bulk_kg)
                truck.timecritical_inventory_kg_current = float(final_tc_kg)
                truck.truck_inventory_kg_current = float(final_body_kg)
                truck.normal_supply_units = int(
                    np.floor(final_bulk_kg / max(self._bulk_supply_unit_kg(), 1e-6))
                )
                truck.emergency_supply_units = int(
                    np.floor(final_tc_kg / max(self._timecritical_supply_unit_kg(), 1e-6))
                )
                truck.cargo = float(final_body_kg / max(float(self.cfg.cargo_unit_kg), 1e-6))

        return JointState(
            step_index=0,
            agents=agents,
            tasks=tasks,
            hazard=HazardSnapshot(
                blocked_ratio=float(self.topology.blocked_ratio()),
                epicenter_node=int(getattr(self.hazards, "epicenter_node", 0)),
            ),
            done=False,
        )

    def reset(self, seed: Optional[int] = None) -> JointState:
        run_seed = int(self.cfg.seed if seed is None else int(seed))
        self.cfg = EnvConfig(**{**self.cfg.__dict__, "seed": run_seed})
        self.rng = np.random.default_rng(self.cfg.seed)
        self.topology = GraphTopology.build_from_config(self.cfg)
        self.hazards = DynamicHazardField(
            topo=self.topology,
            seed=self.cfg.seed + 1,
            stochastic_weather=self.cfg.stochastic_weather,
            cfg=self.cfg,
        )
        self.task_manager = DynamicTaskManager(self.topology)
        self.state = self._build_initial_state()
        self._init_comm_blackout_protocol()
        init_physical_v2_runtime(self)
        self.comm_blocked = {aid: False for aid in self.state.agents}
        self.comm_blackout_agent_observation_count_total = 0
        self.comm_blackout_agent_blocked_count_total = 0
        self.comm_blackout_physical_zone_count_total = 0
        self.comm_blackout_goal_zone_count_total = 0
        self._comm_blackout_active_zone_count = 0
        self.follow_bind_count_total = 0
        self.follow_steps_total = 0
        self.follow_charge_energy_total = 0.0
        self.low_battery_events_total = 0
        self.low_battery_return_success_total = 0
        self.uav_energy_used_total = 0.0
        self.crash_count_total = 0
        self.battery_depletion_count_total = 0
        self.invalid_action_count_total = 0
        self.invalid_action_count_uav_total = 0
        self.invalid_action_count_truck_total = 0
        self.forced_rth_count_total = 0
        self.queue_wait_steps_total = 0
        self._uav_low_battery_flag = {
            aid: False for aid, s in self.state.agents.items() if s.kind == AgentKind.UAV
        }
        self._uav_forced_rth_latch = {
            aid: False for aid, s in self.state.agents.items() if s.kind == AgentKind.UAV
        }
        self._recommended_goals = {}
        self._effective_goals = {}
        # B-only route-stability bookkeeping.  A goal is held only while the
        # underlying routine task remains valid and reachable; hard invalidity
        # and emergency/support goals always bypass the hold.
        self._b_route_stability_goal_step = {}
        self._uav_sortie_contract_task: Dict[str, str] = {}
        self._uav_sortie_contract_version: Dict[str, int] = {}
        self._uav_sortie_recovery_suspended: set[str] = set()
        self._pbrs_lock = {aid: (None, None) for aid in self.state.agents}
        self._pbrs_switch_total = 0
        self._uav_emergency_snap_pending = {}
        self._uav_discovered_blocked_edges = set()
        self._uav_discovered_blocked_total = 0
        self.uav_delivered_tasks_total = 0
        self.uav_delivered_emergency_total = 0
        self.truck_delivered_tasks_total = 0
        self.sortie_limit_hit_total = 0
        self.wind_failure_event_total = 0
        self.wind_failure_risk_accum_total = 0.0
        self.triggered_replans_total = 0
        self.triggered_replans_step = 0
        self.last_assignment_summary = {"assigned_total": 0, "assigned_truck": 0, "assigned_uav": 0}
        self.truck_replenish_count_total = 0
        self.truck_empty_trip_count_total = 0
        self.uav_reload_count_total = 0
        self.uav_reload_wait_steps_total = 0
        self.uav_empty_flight_count_total = 0
        self.uav_delivery_count_total = 0
        self.normal_tasks_blocked_by_supply_count = 0
        self.emergency_tasks_blocked_by_supply_count = 0
        self.uav_recharge_count_total = 0
        self.uav_safe_launch_count_total = 0
        self.uav_launch_count_total = 0
        self.uav_launch_battery_fraction_sum = 0.0
        self.uav_launch_battery_fraction_min = 1.0
        self.uav_unsafe_launch_attempt_count_total = 0
        self.uav_unsafe_launch_block_count_total = 0
        self.uav_low_battery_illegal_launch_count_total = 0
        self.uav_forced_recovery_count_total = 0
        self.uav_forced_recovery_due_to_low_battery_count_total = 0
        self.uav_terminal_battery_rescue_count_total = 0
        self._uav_terminal_battery_rescue_active: set[str] = set()
        self.uav_rendezvous_success_count_total = 0
        self.uav_rendezvous_fail_count_total = 0
        self.truck_recovery_support_count_total = 0
        self.uav_task_reject_below_launch_min_count = 0
        self.uav_task_reject_not_loaded_count = 0
        self.uav_task_reject_recovery_margin_count = 0
        self.uav_task_reject_horizon_count = 0
        self.uav_task_reject_comm_block_count = 0
        self.uav_task_reject_corridor_count = 0
        self.uav_launch_direct_safe_count = 0
        self.uav_launch_rendezvous_safe_count = 0
        self.uav_launch_rendezvous_safe_relaxed_count = 0
        self.uav_launch_block_unsafe_count = 0
        self.uav_launch_gate_enter_count = 0
        self.uav_launch_gate_direct_safe_count = 0
        self.uav_launch_gate_rendezvous_safe_count = 0
        self.uav_launch_gate_rendezvous_safe_relaxed_count = 0
        self.uav_launch_gate_block_below_launch_min_count = 0
        self.uav_launch_gate_block_recovery_margin_count = 0
        self.uav_launch_gate_block_corridor_count = 0
        self.uav_launch_gate_block_other_count = 0
        self.truck_emergency_blocked_by_normal_guard_count = 0
        self.truck_emergency_relief_override_count = 0
        self.truck_emergency_serviceable_count = 0
        self.truck_emergency_not_serviceable_count = 0
        self.island_task_candidate_count = 0
        self.island_task_serviceable_count = 0
        self.island_task_launch_block_count = 0
        self._diag_uav_task_reject_seen_step = set()
        self._diag_truck_emergency_relief_seen_step = set()
        self._diag_truck_emergency_serviceability_seen_step = set()
        self._island_task_ids_seen = set()
        self.island_task_completed_count_total = 0
        self.truck_forward_support_count_total = 0
        self.truck_forward_support_distance_total = 0.0
        self.truck_uav_assist_waypoint_move_count_total = 0
        self._planner_truck_assist_waypoint_by_truck = {}
        self._planner_route_plan_v2 = {}
        self._uav_sortie_contract_task = {}
        self._uav_sortie_contract_version = {}
        self._uav_sortie_recovery_suspended = set()
        self.uav_authoritative_sortie_goal_override_count = 0
        self.uav_terminal_delivery_commitment_count = 0
        self._planner_route_plan_feedback = []
        self._planner_route_plan_stay_reason_by_agent = {}
        self._planner_route_plan_goals = {}
        # Execution-layer counters for atomic onsite routine capture.  These are
        # deliberately separate from planner-level takeover counters so paper
        # diagnostics can distinguish real service protection from route rebinding.
        self.route_plan_v2_onsite_capture_count = 0
        self.route_plan_v2_onsite_capture_contract_transfer_count = 0
        self.route_plan_v2_onsite_capture_preempted_assist_count = 0
        self._erc_v2_command_gate_enabled = False
        self._erc_v2_command_batch = None
        self.unauthorized_support_attempt_count = 0
        self.unauthorized_support_blocked_count = 0
        self.unauthorized_recovery_attempt_count = 0
        self.unauthorized_recovery_blocked_count = 0
        self.command_rejected_count = 0
        self.command_rejected_reason_launch_unauthorized_count = 0
        self.support_command_count = 0
        self.support_command_to_launch_count = 0
        self.support_command_to_delivery_count = 0
        self.safety_recovery_command_count = 0
        self.routine_near_completion_protected_count = 0
        self.routine_near_completion_support_blocked_count = 0
        self.routine_near_completion_recovery_blocked_count = 0
        self.routine_near_completion_broken_by_hard_safety_count = 0
        self.routine_near_completion_broken_by_tc_override_count = 0
        self.routine_near_completion_tc_override_to_launch_count = 0
        self.routine_near_completion_tc_override_to_delivery_count = 0
        self.routine_near_completion_blocked_tc_support_count = 0
        self.routine_near_completion_followed_by_service_start_count = 0
        self.routine_near_completion_followed_by_completion_count = 0
        self.routine_near_completion_tc_override_reject_delay_count = 0
        self.routine_near_completion_tc_override_reject_no_loaded_uav_count = 0
        self.routine_near_completion_tc_override_reject_no_candidate_count = 0
        self.routine_near_completion_tc_override_reject_not_near_launchable_count = 0
        self.routine_near_completion_tc_override_reject_recovery_count = 0
        self.routine_near_completion_broken_by_delivery_feasible_tc_override_count = 0
        self.routine_multiround_commitment_count = 0
        self.routine_multiround_support_block_count = 0
        self._routine_multiround_service_commitment_by_truck = {}
        self.tc_override_candidate_count = 0
        self.tc_override_blocked_not_full_sortie_feasible_count = 0
        self.tc_override_blocked_low_recovery_margin_count = 0
        self.tc_override_blocked_low_battery_margin_count = 0
        self.tc_override_blocked_recent_reject_count = 0
        self.tc_override_blocked_lifeline_risk_count = 0
        self.tc_override_blocked_routine_delay_count = 0
        self.tc_override_to_launch_count = 0
        self.tc_override_to_delivery_count = 0
        self.tc_override_to_forced_recovery_count = 0
        self.tc_override_feasibility_mismatch_count = 0
        self.tc_override_predicted_launchable_count = 0
        self.tc_override_actual_launch_count = 0
        self.tc_override_predicted_delivery_feasible_count = 0
        self.tc_override_actual_delivery_count = 0
        self._tc_override_recent_tasks = {}
        self._tc_override_trace_rows = []
        self._tc_override_recent_reject = {}
        self._routine_protection_recent_tasks = {}
        self._routine_tc_override_recent_tasks = {}
        self.uav_island_delivery_count_total = 0
        self.uav_island_recovery_success_count_total = 0
        self.relaxed_sortie_selected_count_total = 0
        self.relaxed_delivery_completed_count_total = 0
        self.uav_docked_retarget_count_total = 0
        self.uav_docked_retarget_count_step = 0
        self.uav_urgent_watchdog_assign_count_total = 0
        self.uav_urgent_watchdog_assign_count_step = 0
        self._uav_last_docked_retarget_step = {
            aid: -10**9 for aid, s in self.state.agents.items() if s.kind == AgentKind.UAV
        }
        self._uav_docked_goal_hold_steps = {
            aid: 0 for aid, s in self.state.agents.items() if s.kind == AgentKind.UAV
        }
        self._uav_docked_goal_hold_task = {
            aid: None for aid, s in self.state.agents.items() if s.kind == AgentKind.UAV
        }
        self._uav_post_bind_dwell_remaining = {
            aid: (
                int(max(getattr(self.cfg, "uav_post_bind_min_dwell_steps", 0), 0))
                if getattr(s, "follow_target", None) is not None
                else 0
            )
            for aid, s in self.state.agents.items()
            if s.kind == AgentKind.UAV
        }
        self._uav_bind_commit_target = {
            aid: None for aid, s in self.state.agents.items() if s.kind == AgentKind.UAV
        }
        self._uav_bind_commit_until_step = {
            aid: -1 for aid, s in self.state.agents.items() if s.kind == AgentKind.UAV
        }
        self._uav_forced_rth_start_step = {
            aid: -1 for aid, s in self.state.agents.items() if s.kind == AgentKind.UAV
        }
        self._uav_recovery_requested_truck = {}
        self._uav_transfer_target_truck = {}
        self._uav_transfer_target_task = {}
        self._uav_post_transfer_contract_task = {}
        self._planner_truck_assist_waypoint_by_truck = {}
        self._planner_route_plan_v2 = {}
        self.uav_authoritative_sortie_goal_override_count = 0
        self.uav_terminal_delivery_commitment_count = 0
        self._planner_route_plan_feedback = []
        self._planner_route_plan_stay_reason_by_agent = {}
        self._planner_route_plan_goals = {}
        self._uav_last_launch_reason = {
            aid: "" for aid, s in self.state.agents.items() if s.kind == AgentKind.UAV
        }
        self._uav_sortie_relaxed_latch = {
            aid: False for aid, s in self.state.agents.items() if s.kind == AgentKind.UAV
        }
        self.initial_route_docked_uav_count = int(
            sum(1 for _aid, _s in self.state.agents.items() if _s.kind == AgentKind.UAV and getattr(_s, "follow_target", None) is not None)
        )
        self._initial_route_docked_truck_by_uav = {
            str(aid): str(getattr(s, "follow_target", ""))
            for aid, s in self.state.agents.items()
            if s.kind == AgentKind.UAV and getattr(s, "follow_target", None) is not None
        }
        self._uav_launch_block_cooldown_until_step = {
            aid: -1 for aid, s in self.state.agents.items() if s.kind == AgentKind.UAV
        }
        self.planner_replan_due_to_new_road_info_count_total = 0
        self.planner_refresh_map_update_step = False
        self.planner_last_replan_reason = "none"
        self._cached_island_task_ids_token = None
        self._cached_island_task_ids = set()
        self._decision_sp_cache_token = None
        self._decision_sp_cache = {}
        self._truck_last_arrived_from = {}
        self._init_shared_road_awareness_state()
        return self.state

    def _bulk_supply_unit_kg(self) -> float:
        return float(max(float(getattr(self.cfg, "bulk_supply_unit_kg", 300.0)), 1e-6))

    def _timecritical_supply_unit_kg(self) -> float:
        return float(max(float(getattr(self.cfg, "timecritical_supply_unit_kg", 150.0)), 1e-6))

    def _truck_initial_bulk_inventory_kg(self) -> float:
        bulk_kg = float(getattr(self.cfg, "truck_initial_bulk_inventory_kg", 0.0))
        if bulk_kg > 0.0:
            return float(bulk_kg)
        return float(max(int(getattr(self.cfg, "truck_initial_normal_supply_units", 0)), 0) * float(self.cfg.normal_task_demand_kg))

    def _truck_initial_timecritical_inventory_kg(self) -> float:
        tc_kg = float(getattr(self.cfg, "truck_initial_timecritical_inventory_kg", 0.0))
        if tc_kg > 0.0:
            return float(tc_kg)
        # Backward-compatible fallback from legacy emergency units.
        legacy_unit_kg = float(max(self._timecritical_supply_unit_kg(), float(self.cfg.emergency_task_demand_kg)))
        return float(max(int(getattr(self.cfg, "truck_initial_emergency_supply_units", 0)), 0) * legacy_unit_kg)

    def _truck_full_inventory_kg(self) -> float:
        return float(self._truck_initial_bulk_inventory_kg() + self._truck_initial_timecritical_inventory_kg())

    def _mounted_uav_payload_kg_by_supply_type(
        self,
        truck_id: str,
        *,
        exclude_uav_id: Optional[str] = None,
    ) -> Tuple[float, float]:
        """Return normal/emergency material currently mounted on one truck.

        UAV airframe mass is deliberately excluded: the 3000 kg contract is a
        material-load contract, while ``uav_self_weight_kg`` documents the
        aircraft's own 50 kg mass separately.
        """
        bulk_kg = 0.0
        tc_kg = 0.0
        state = getattr(self, "state", None)
        if state is None:
            return bulk_kg, tc_kg
        for agent_id, agent in state.agents.items():
            if agent.kind != AgentKind.UAV or str(getattr(agent, "follow_target", "")) != str(truck_id):
                continue
            if exclude_uav_id is not None and str(agent_id) == str(exclude_uav_id):
                continue
            payload_kg = float(max(getattr(agent, "payload_kg_current", 0.0), 0.0))
            if str(getattr(agent, "payload_supply_type", "emergency")).strip().lower() == "normal":
                bulk_kg += payload_kg
            else:
                tc_kg += payload_kg
        return float(bulk_kg), float(tc_kg)

    def _truck_material_load_with_mounted_uavs_kg(
        self,
        truck_id: str,
        *,
        exclude_uav_id: Optional[str] = None,
    ) -> float:
        truck = self.state.agents.get(str(truck_id), None)
        if truck is None or truck.kind != AgentKind.TRUCK:
            return float("inf")
        mounted_bulk_kg, mounted_tc_kg = self._mounted_uav_payload_kg_by_supply_type(
            str(truck_id), exclude_uav_id=exclude_uav_id
        )
        return float(
            max(getattr(truck, "truck_inventory_kg_current", 0.0), 0.0)
            + mounted_bulk_kg
            + mounted_tc_kg
        )

    def _truck_can_accept_uav_payload(self, truck_id: str, uav_id: str) -> bool:
        uav = self.state.agents.get(str(uav_id), None)
        if uav is None or uav.kind != AgentKind.UAV:
            return False
        current_material_kg = self._truck_material_load_with_mounted_uavs_kg(
            str(truck_id), exclude_uav_id=str(uav_id)
        )
        incoming_payload_kg = float(max(getattr(uav, "payload_kg_current", 0.0), 0.0))
        capacity_kg = float(max(getattr(self.cfg, "truck_payload_capacity_kg", 0.0), 0.0))
        return bool(current_material_kg + incoming_payload_kg <= capacity_kg + 1e-9)

    def _truck_transport_mass_kg(self, truck_id: str) -> float:
        """Gross carried mass affecting speed; airframes do not consume cargo capacity."""
        material_kg = self._truck_material_load_with_mounted_uavs_kg(str(truck_id))
        if not np.isfinite(material_kg):
            return 0.0
        mounted_uav_count = sum(
            1
            for agent in self.state.agents.values()
            if agent.kind == AgentKind.UAV
            and str(getattr(agent, "follow_target", "")) == str(truck_id)
        )
        return float(
            material_kg
            + float(mounted_uav_count)
            * float(max(getattr(self.cfg, "uav_self_weight_kg", 50.0), 0.0))
        )

    def _truck_replenish_inventory_targets(self, truck_id: str) -> Tuple[float, float]:
        """Truck-body refill targets that include mounted UAV cargo only once."""
        mounted_bulk_kg, mounted_tc_kg = self._mounted_uav_payload_kg_by_supply_type(str(truck_id))
        target_bulk_kg = float(max(self._truck_initial_bulk_inventory_kg() - mounted_bulk_kg, 0.0))
        target_tc_kg = float(max(self._truck_initial_timecritical_inventory_kg() - mounted_tc_kg, 0.0))
        body_capacity_kg = float(
            max(
                float(getattr(self.cfg, "truck_payload_capacity_kg", self._truck_full_inventory_kg()))
                - mounted_bulk_kg
                - mounted_tc_kg,
                0.0,
            )
        )
        if target_bulk_kg + target_tc_kg > body_capacity_kg + 1e-9:
            # Preserve routine bulk first; emergency stock uses the remaining
            # physical capacity because mounted UAV emergency packages already
            # count toward the cooperative unit's four-package quota.
            target_bulk_kg = float(min(target_bulk_kg, body_capacity_kg))
            target_tc_kg = float(min(target_tc_kg, max(body_capacity_kg - target_bulk_kg, 0.0)))
        return float(target_bulk_kg), float(target_tc_kg)

    def _task_supply_type(self, task: DeliveryTask) -> str:
        if str(getattr(task, "supply_type", "")).strip().lower() in {"normal", "emergency"}:
            return str(task.supply_type).strip().lower()
        return "emergency" if task.kind == TaskKind.EMERGENCY else "normal"

    def _task_class(self, task: DeliveryTask) -> str:
        tc = str(getattr(task, "task_class", "")).strip().lower()
        if tc in {TaskClass.ROUTINE_BULK.value, TaskClass.TIME_CRITICAL_LIGHTWEIGHT.value}:
            return tc
        return TaskClass.TIME_CRITICAL_LIGHTWEIGHT.value if task.kind == TaskKind.EMERGENCY else TaskClass.ROUTINE_BULK.value

    def _task_is_routine_bulk(self, task: DeliveryTask) -> bool:
        return bool(self._task_class(task) == TaskClass.ROUTINE_BULK.value)

    def _task_is_time_critical_lightweight(self, task: DeliveryTask) -> bool:
        return bool(self._task_class(task) == TaskClass.TIME_CRITICAL_LIGHTWEIGHT.value)

    def _task_is_bulk_relay(self, task: Optional[DeliveryTask]) -> bool:
        """Return whether a road-isolated normal task uses UAV relay service.

        The task remains NORMAL/routine_bulk for every paper metric; only its
        execution mode changes.
        """
        if task is None:
            return False
        return bool(
            task.kind == TaskKind.NORMAL
            and str(getattr(task, "service_mode", "DIRECT")).strip().upper()
            == "BULK_RELAY"
        )

    def _task_is_uav_delivery(self, task: Optional[DeliveryTask]) -> bool:
        if task is None:
            return False
        return bool(task.kind == TaskKind.EMERGENCY or self._task_is_bulk_relay(task))

    def _task_supply_units_required(self, task: DeliveryTask) -> int:
        return int(max(1, int(getattr(task, "supply_units_required", 1))))

    def _task_demand_units(self, task: DeliveryTask) -> int:
        return int(max(1, int(getattr(task, "demand_units", 1))))

    def _task_demand_kg(self, task: DeliveryTask) -> float:
        val = float(getattr(task, "demand_kg", 0.0))
        if val > 0.0:
            return val
        if self._task_supply_type(task) == "emergency":
            return float(self.cfg.emergency_task_demand_kg)
        return float(self.cfg.normal_task_demand_kg)

    def _has_pending_supply_type_tasks(self, supply_type: str) -> bool:
        stype = str(supply_type).strip().lower()
        for task in self.state.tasks.values():
            if task.status not in (TaskStatus.PENDING, TaskStatus.CLAIMED):
                continue
            if self._task_supply_type(task) == stype:
                return True
        return False

    def _truck_needs_replenish_for_pending_tasks(self, s: AgentRuntimeState) -> bool:
        if s.kind != AgentKind.TRUCK:
            return False
        bulk_kg = float(max(getattr(s, "bulk_inventory_kg_current", 0.0), 0.0))
        tc_kg = float(max(getattr(s, "timecritical_inventory_kg_current", 0.0), 0.0))
        need_normal = bool(self._has_pending_supply_type_tasks("normal"))
        # User-required policy: truck emergency stock-out should not force depot by default.
        need_emergency = bool(
            bool(getattr(self.cfg, "truck_replenish_for_emergency_stock", False))
            and self._has_pending_supply_type_tasks("emergency")
        )
        tc_unit = float(self._timecritical_supply_unit_kg())
        return bool((need_normal and bulk_kg <= 1e-9) or (need_emergency and tc_kg < tc_unit - 1e-9))

    def _sync_truck_inventory_fields(self, s: AgentRuntimeState) -> None:
        if s.kind != AgentKind.TRUCK:
            return
        bulk_kg = float(max(getattr(s, "bulk_inventory_kg_current", 0.0), 0.0))
        tc_kg = float(max(getattr(s, "timecritical_inventory_kg_current", 0.0), 0.0))

        # Backward-compatible bootstrap when loading old state/checkpoint shapes.
        if bulk_kg <= 1e-9 and tc_kg <= 1e-9:
            legacy_n = int(max(getattr(s, "normal_supply_units", 0), 0))
            legacy_e = int(max(getattr(s, "emergency_supply_units", 0), 0))
            if legacy_n > 0 or legacy_e > 0:
                bulk_kg = float(legacy_n) * float(self.cfg.normal_task_demand_kg)
                tc_kg = float(legacy_e) * float(max(self._timecritical_supply_unit_kg(), float(self.cfg.emergency_task_demand_kg)))

        s.bulk_inventory_kg_current = float(bulk_kg)
        s.timecritical_inventory_kg_current = float(tc_kg)

        bulk_unit = float(self._bulk_supply_unit_kg())
        tc_unit = float(self._timecritical_supply_unit_kg())
        s.normal_supply_units = int(np.floor(float(bulk_kg) / max(bulk_unit, 1e-6)))
        s.emergency_supply_units = int(np.floor(float(tc_kg) / max(tc_unit, 1e-6)))
        s.truck_inventory_kg_current = float(bulk_kg + tc_kg)

        # Backward-compat cargo proxy (abstract units) retained for older components.
        s.cargo = float(s.truck_inventory_kg_current / max(float(self.cfg.cargo_unit_kg), 1e-6))
        # Replenish only when a required supply-type is depleted and pending tasks still need it.
        s.truck_needs_replenish_flag = bool(self._truck_needs_replenish_for_pending_tasks(s))

    def _sync_uav_payload_fields(self, s: AgentRuntimeState) -> None:
        if s.kind != AgentKind.UAV:
            return
        s.carried_emergency_units = int(
            np.clip(int(getattr(s, "carried_emergency_units", 0)), 0, int(max(self.cfg.uav_max_emergency_units, 1)))
        )
        if s.carried_emergency_units <= 0:
            s.payload_kg_current = 0.0
            s.cargo = 0.0
        else:
            if float(getattr(s, "payload_kg_current", 0.0)) <= 0.0:
                s.payload_kg_current = float(self._timecritical_supply_unit_kg())
            s.cargo = float(self.cfg.uav_cargo_capacity_units)

    def _uav_loaded(self, aid: str) -> bool:
        s = self.state.agents[str(aid)]
        if s.kind != AgentKind.UAV:
            return False
        self._sync_uav_payload_fields(s)
        return bool(
            int(getattr(s, "carried_emergency_units", 0)) >= 1
            and float(getattr(s, "payload_kg_current", 0.0)) >= float(self.cfg.emergency_task_demand_kg) - 1e-9
            and (not bool(getattr(s, "uav_needs_reload_flag", False)))
        )

    def _uav_loaded_for_task(self, aid: str, task: Optional[DeliveryTask]) -> bool:
        if not self._uav_loaded(str(aid)):
            return False
        expected = "normal" if self._task_is_bulk_relay(task) else "emergency"
        actual = str(
            getattr(self.state.agents[str(aid)], "payload_supply_type", "emergency")
        ).strip().lower()
        return bool(actual == expected)

    def _uav_relay_payload_capacity_kg(self, task: Optional[DeliveryTask]) -> float:
        """Return the physical payload cap for either UAV delivery mode."""
        if self._task_is_bulk_relay(task):
            return float(
                min(
                    max(getattr(self.cfg, "hrl_route_plan_bulk_relay_payload_kg", 40.0), 1e-6),
                    max(getattr(self.cfg, "uav_payload_capacity_kg", 40.0), 1e-6),
                )
            )
        return float(max(getattr(self.cfg, "uav_payload_capacity_kg", self._timecritical_supply_unit_kg()), 1e-6))

    def _truck_follower_count(self, truck_id: str, exclude_aid: Optional[str] = None) -> int:
        return int(
            sum(
                1
                for uid, us in self.state.agents.items()
                if us.kind == AgentKind.UAV
                and (exclude_aid is None or str(uid) != str(exclude_aid))
                and us.follow_target is not None
                and str(us.follow_target) == str(truck_id)
            )
        )

    def _truck_has_follow_slot(self, truck_id: str, exclude_aid: Optional[str] = None) -> bool:
        cap = int(max(getattr(self.cfg, "uav_max_followers_per_truck", 0), 0))
        if cap <= 0:
            return False
        return bool(self._truck_follower_count(str(truck_id), exclude_aid=exclude_aid) < cap)

    def _nearest_truck_from_xy(
        self,
        xy: Tuple[float, float],
        require_emergency_stock: bool = False,
        require_follow_slot: bool = False,
        exclude_aid: Optional[str] = None,
    ) -> Tuple[Optional[str], float]:
        x, y = float(xy[0]), float(xy[1])
        best_tid: Optional[str] = None
        best_d = float("inf")
        for tid, ts in self.state.agents.items():
            if ts.kind != AgentKind.TRUCK or bool(getattr(ts, "crashed", False)):
                continue
            self._sync_truck_inventory_fields(ts)
            if require_emergency_stock and int(getattr(ts, "emergency_supply_units", 0)) <= 0:
                continue
            if require_follow_slot and (not self._truck_has_follow_slot(str(tid), exclude_aid=exclude_aid)):
                continue
            if (
                require_follow_slot
                and exclude_aid is not None
                and (not self._truck_can_accept_uav_payload(str(tid), str(exclude_aid)))
            ):
                continue
            txy = ts.pos_xy if ts.pos_xy is not None else self._node_xy(int(ts.node or 0))
            d = float(np.hypot(x - float(txy[0]), y - float(txy[1])))
            if d < best_d:
                best_d = d
                best_tid = str(tid)
        return best_tid, best_d

    def _any_truck_with_emergency_stock(self) -> bool:
        for ts in self.state.agents.values():
            if ts.kind != AgentKind.TRUCK or bool(getattr(ts, "crashed", False)):
                continue
            self._sync_truck_inventory_fields(ts)
            if int(getattr(ts, "emergency_supply_units", 0)) > 0:
                return True
        return False

    def _uav_needs_reload(self, aid: str) -> bool:
        s = self.state.agents[str(aid)]
        if s.kind != AgentKind.UAV:
            return False
        return bool(
            bool(getattr(s, "uav_needs_reload_flag", False))
            or int(getattr(s, "carried_emergency_units", 0)) < int(max(self.cfg.uav_max_emergency_units, 1))
            or float(getattr(s, "payload_kg_current", 0.0)) < float(self._timecritical_supply_unit_kg()) - 1e-9
        )

    def _uav_should_seek_depot(self, aid: str) -> bool:
        if not bool(getattr(self.cfg, "uav_reload_at_depot_enabled", True)):
            return False
        s = self.state.agents[str(aid)]
        if s.kind != AgentKind.UAV:
            return False
        needs_reload = bool(self._uav_needs_reload(str(aid)))
        if needs_reload and (not self._any_truck_with_emergency_stock()):
            return True
        return False

    def _uav_try_dock_depot(self, aid: str) -> bool:
        s = self.state.agents[str(aid)]
        if s.kind != AgentKind.UAV or s.follow_target is not None:
            return False
        if not bool(getattr(self.cfg, "uav_reload_at_depot_enabled", True)):
            return False
        # Depot auto-dock is used as reload fallback (UAV out of emergency payload),
        # not as a general low-battery teleport guard.
        if not bool(self._uav_needs_reload(str(aid))):
            return False
        ux, uy = s.pos_xy if s.pos_xy is not None else self._node_xy(int(s.node or 0))
        dx, dy = self._node_xy(0)
        dist = float(np.hypot(float(ux) - float(dx), float(uy) - float(dy)))
        dock_radius = float(max(getattr(self.cfg, "uav_bind_radius_m", 170.0), 1.0))
        if dist > dock_radius:
            return False
        s.follow_target = DEPOT_DOCK_ID
        s.node = 0
        s.pos_xy = self._node_xy(0)
        s.vel_xy = (0.0, 0.0)
        s.sortie_distance_m = 0.0
        s.replenish_timer = int(max(0, int(self.cfg.replenish_freeze_steps)))
        return True

    def get_payload_factor(self, payload_kg: float) -> float:
        capacity = float(max(getattr(self.cfg, "uav_payload_capacity_kg", 1.0), 1e-6))
        ratio = float(np.clip(float(payload_kg) / capacity, 0.0, 1.0))
        penalty = float(max(getattr(self.cfg, "uav_full_load_energy_penalty", 0.0), 0.0))
        return float(1.0 + penalty * ratio)

    def calculate_uav_leg_energy(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        payload_kg: float,
    ) -> float:
        dx = float(destination[0]) - float(origin[0])
        dy = float(destination[1]) - float(origin[1])
        distance = float(np.hypot(dx, dy))
        if distance <= 1e-9:
            return 0.0
        weather = self.hazards.weather_at((float(origin[0]), float(origin[1])))
        wind_x, wind_y = self.hazards.wind_vector_at((float(origin[0]), float(origin[1])))
        headwind = float(max(0.0, -(float(wind_x) * dx + float(wind_y) * dy) / distance))
        weather_factor = float(
            1.0
            + float(max(getattr(self.cfg, "uav_headwind_energy_coeff", 0.0), 0.0)) * headwind
            + float(max(getattr(self.cfg, "uav_rain_energy_coeff", 0.0), 0.0)) * max(float(weather.rain), 0.0)
        )
        return float(
            distance
            * float(max(getattr(self.cfg, "uav_flight_discharge_per_m", 0.0), 0.0))
            * weather_factor
            * self.get_payload_factor(payload_kg)
        )

    def _uav_energy_cost_fraction(
        self,
        aid: str,
        dist_m: float,
        xy: Tuple[float, float],
        *,
        destination: Optional[Tuple[float, float]] = None,
        payload_override: Optional[float] = None,
    ) -> float:
        """Return one leg's battery fraction using one canonical geometry path.

        ``destination`` is optional for backwards compatibility with older
        planners.  Runtime movement and new physical predictions must provide
        it so headwind is evaluated along the actual old->new direction rather
        than the historical synthetic ``+x`` direction.  ``payload_override``
        is used for the post-service recovery leg, whose payload is normally
        empty even though the UAV was loaded at launch.
        """
        s = self.state.agents[str(aid)]
        if s.kind != AgentKind.UAV:
            return float("inf")
        payload = float(
            max(
                getattr(s, "payload_kg_current", 0.0)
                if payload_override is None
                else payload_override,
                0.0,
            )
        )
        v2_fraction = v2_uav_energy_cost_fraction(self, str(aid), float(dist_m), xy)
        if v2_fraction is not None:
            return float(v2_fraction)
        origin = (float(xy[0]), float(xy[1]))
        if destination is None:
            destination = (float(xy[0]) + float(max(dist_m, 0.0)), float(xy[1]))
        return self.calculate_uav_leg_energy(origin, destination, payload)

    def _uav_weather_safety_reason(
        self,
        xy: Tuple[float, float],
        *,
        recovery: bool = False,
    ) -> str:
        """Return the canonical V1 weather safety reason for a position."""
        weather = self.hazards.weather_at((float(xy[0]), float(xy[1])))
        wind = float(max(getattr(weather, "wind", 0.0), 0.0))
        rain = float(max(getattr(weather, "rain", 0.0), 0.0))
        wind_limit = float(
            max(
                getattr(
                    self.cfg,
                    "uav_recover_wind_mps" if recovery else "uav_no_launch_wind_mps",
                    10.0 if recovery else 6.0,
                ),
                0.0,
            )
        )
        rain_limit = float(
            max(
                getattr(
                    self.cfg,
                    "uav_recover_rain_mmh" if recovery else "uav_no_launch_rain_mmh",
                    30.0 if recovery else 24.0,
                ),
                0.0,
            )
        )
        if wind >= wind_limit - 1e-9:
            return "unsafe_wind"
        if rain >= rain_limit - 1e-9:
            return "unsafe_rain"
        return ""

    @staticmethod
    def _uav_extended_destination(
        origin: Tuple[float, float],
        target: Tuple[float, float],
        distance_m: float,
    ) -> Tuple[float, float]:
        """Extend a leg toward ``target`` when a safety buffer is added."""
        ox, oy = float(origin[0]), float(origin[1])
        tx, ty = float(target[0]), float(target[1])
        requested = float(max(distance_m, 0.0))
        dx, dy = tx - ox, ty - oy
        norm = float(np.hypot(dx, dy))
        if norm <= 1e-9:
            return ox + requested, oy
        scale = requested / norm
        return ox + dx * scale, oy + dy * scale

    def _uav_actual_movement_energy(
        self,
        aid: str,
        old_xy: Tuple[float, float],
        new_xy: Tuple[float, float],
    ) -> float:
        """Debit/estimate a movement leg with its actual direction."""
        dx = float(new_xy[0]) - float(old_xy[0])
        dy = float(new_xy[1]) - float(old_xy[1])
        distance = float(np.hypot(dx, dy))
        try:
            return self._uav_energy_cost_fraction(
                str(aid),
                distance,
                (float(old_xy[0]), float(old_xy[1])),
                destination=(float(new_xy[0]), float(new_xy[1])),
            )
        except TypeError as exc:
            # Keep compatibility with focused tests/legacy adapters that wrap
            # the old three-argument estimator while the runtime uses the
            # direction-aware signature.
            if "destination" not in str(exc):
                raise
            return self._uav_energy_cost_fraction(
                str(aid), distance, (float(old_xy[0]), float(old_xy[1]))
            )

    def _uav_expected_payload_after_task(self, aid: str, task: Optional[DeliveryTask]) -> float:
        """Predict the physical payload left after servicing ``task``."""
        s = self.state.agents.get(str(aid))
        if s is None or s.kind != AgentKind.UAV:
            return 0.0
        current = float(max(getattr(s, "payload_kg_current", 0.0), 0.0))
        if task is None:
            return current
        demand = float(max(getattr(task, "remaining_demand_kg", 0.0), 0.0))
        return float(max(current - demand, 0.0))

    def _uav_sortie_energy_requirement(
        self,
        aid: str,
        task: Optional[DeliveryTask],
        *,
        recovery_buffer_m: float = 0.0,
    ) -> float:
        """Estimate loaded outbound plus empty/residual recovery energy.

        This helper is intentionally pure: it does not mutate battery or any
        physical ledger and is safe for planner feasibility checks.
        """
        s = self.state.agents.get(str(aid))
        if s is None or s.kind != AgentKind.UAV or task is None:
            return float("inf")
        cur_xy = s.pos_xy if s.pos_xy is not None else self._node_xy(int(s.node or 0))
        task_xy = self._node_xy(int(task.demand_node))
        _, d_back = self._nearest_truck_from_xy(task_xy)
        if not np.isfinite(d_back):
            return float("inf")
        truck_id, _ = self._nearest_truck_from_xy(task_xy)
        truck_xy = task_xy
        if truck_id is not None and truck_id in self.state.agents:
            truck = self.state.agents[str(truck_id)]
            truck_xy = truck.pos_xy if truck.pos_xy is not None else self._node_xy(int(truck.node or 0))
        payload = float(max(getattr(s, "payload_kg_current", 0.0), 0.0))
        req_go = self._uav_energy_cost_fraction(
            str(aid),
            float(np.hypot(float(task_xy[0]) - float(cur_xy[0]), float(task_xy[1]) - float(cur_xy[1]))),
            cur_xy,
            destination=task_xy,
            payload_override=payload,
        )
        return_dist = float(max(d_back + float(max(recovery_buffer_m, 0.0)), 0.0))
        req_back = self._uav_energy_cost_fraction(
            str(aid),
            return_dist,
            task_xy,
            destination=self._uav_extended_destination(task_xy, truck_xy, return_dist),
            payload_override=self._uav_expected_payload_after_task(str(aid), task),
        )
        return float(req_go + req_back)

    def _legacy_sortie_cap_enabled(self) -> bool:
        """Enable the V1 operational cumulative-sortie distance envelope."""
        return bool(
            str(getattr(self.cfg, "physical_environment_version", "v1")).lower() != "v2"
            and getattr(self.cfg, "uav_enforce_max_sortie_limit", False)
        )

    def _uav_bind_window_m(self, truck_state: Optional[AgentRuntimeState] = None) -> float:
        base = float(max(getattr(self.cfg, "uav_bind_radius_m", 170.0), 0.0))
        gain = float(max(getattr(self.cfg, "uav_bind_motion_window_gain", 0.30), 0.0))
        latency_s = float(max(getattr(self.cfg, "uav_bind_latency_margin_s", 5.0), 0.0))
        dt = float(max(self._dt_seconds, 0.0))

        truck_speed = float(max(getattr(self.cfg, "truck_speed_mps", 0.0), 0.0))
        ts = truck_state
        if ts is not None and ts.kind == AgentKind.TRUCK and ts.transit is not None:
            src, dst, _ = ts.transit
            payload_kg = float(self._truck_transport_mass_kg(str(getattr(ts, "agent_id", ""))))
            truck_speed = float(max(self._truck_speed_mps(int(src), int(dst), payload_kg=payload_kg), 0.0))

        dynamic_extra = float(gain * truck_speed * dt + truck_speed * latency_s)
        return float(base + max(dynamic_extra, 0.0))

    def _uav_mark_forced_recovery(self, aid: str, reason: str = "general") -> bool:
        if not self._uav_forced_rth_latch.get(str(aid), False):
            self._uav_forced_rth_latch[str(aid)] = True
            self._uav_forced_rth_start_step[str(aid)] = int(self.state.step_index)
            self.uav_forced_recovery_count_total += 1
            if str(reason) == "low_battery":
                self.uav_forced_recovery_due_to_low_battery_count_total += 1
            return True
        self._uav_forced_rth_latch[str(aid)] = True
        self._uav_forced_rth_start_step.setdefault(str(aid), int(self.state.step_index))
        return False

    def _task_deadline_slack_steps(self, task: Optional[DeliveryTask]) -> int:
        if task is None:
            return int(max(int(self.cfg.max_steps) - int(self.state.step_index), 0))
        return int(max(int(getattr(task, "deadline_step", self.cfg.max_steps)) - int(self.state.step_index), 0))

    def _is_l_pressure_context(self) -> bool:
        phase = str(getattr(self.cfg, "phase", "")).strip().upper()
        map_size = float(max(getattr(self.cfg, "map_size_m", 0.0), 0.0))
        node_count = int(max(getattr(self.cfg, "num_nodes", getattr(self.cfg, "n_nodes", 0)), 0))
        return bool(phase == "L" or map_size >= 10000.0 or node_count >= 120)

    def _task_is_island(self, task: Optional[DeliveryTask]) -> bool:
        if task is None or task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
            return False
        return bool(str(task.task_id) in self._current_island_emergency_task_ids())

    def _task_nearest_truck_distance_m(self, task: Optional[DeliveryTask]) -> float:
        if task is None:
            return float("inf")
        node = self.topology.nodes[int(task.demand_node)]
        _, d = self._nearest_truck_from_xy((float(node.x), float(node.y)))
        return float(d)

    def _task_uav_cover_fraction(self, task: Optional[DeliveryTask]) -> float:
        """Pure coverage estimate used by pressure classification.

        This helper must not call launch-threshold helpers or launch-gate helpers.
        """
        if task is None or task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
            return 0.0
        uavs = [
            str(uid)
            for uid, us in self.state.agents.items()
            if us.kind == AgentKind.UAV and (not bool(getattr(us, "crashed", False)))
        ]
        if not uavs:
            return 0.0
        node = self.topology.nodes[int(task.demand_node)]
        txy = (float(node.x), float(node.y))
        back_tid, d_back = self._nearest_truck_from_xy(txy)
        if not np.isfinite(d_back):
            return 0.0

        feasible = 0
        launch_field = "uav_min_takeoff_soc" if str(getattr(self.cfg, "physical_environment_version", "v1")).lower() == "v2" else "uav_launch_min_battery_fraction"
        base_launch_thr = float(np.clip(getattr(self.cfg, launch_field, 0.10), 0.0, 1.0))
        reserve = float(np.clip(getattr(self.cfg, "uav_emergency_reserve_fraction", 0.20), 0.0, 1.0))
        return_margin = float(np.clip(getattr(self.cfg, "uav_return_margin_fraction", 0.15), 0.0, 1.0))
        rendez_margin = float(np.clip(getattr(self.cfg, "uav_rendezvous_margin_fraction", 0.10), 0.0, 1.0))
        recovery_buf = float(max(getattr(self.cfg, "uav_recovery_distance_buffer_m", 600.0), 0.0))
        for uid in uavs:
            us = self.state.agents[str(uid)]
            if bool(getattr(us, "uav_needs_reload_flag", False)) or (not bool(self._uav_loaded(str(uid)))):
                continue
            batt = float(max(getattr(us, "battery", 0.0), 0.0))
            if batt + 1e-9 < base_launch_thr:
                continue
            cur_xy = us.pos_xy if us.pos_xy is not None else self._node_xy(int(us.node or 0))
            d_go = float(np.hypot(float(cur_xy[0]) - txy[0], float(cur_xy[1]) - txy[1]))
            recovery_xy = cur_xy
            if back_tid is not None and str(back_tid) in self.state.agents:
                back_state = self.state.agents[str(back_tid)]
                recovery_xy = back_state.pos_xy if back_state.pos_xy is not None else self._node_xy(int(back_state.node or 0))
            return_dist = float(d_back + recovery_buf)
            req_go = float(
                self._uav_energy_cost_fraction(
                    str(uid), d_go, cur_xy, destination=txy,
                    payload_override=float(getattr(us, "payload_kg_current", 0.0)),
                )
            )
            req_back = float(
                self._uav_energy_cost_fraction(
                    str(uid), return_dist, txy,
                    destination=self._uav_extended_destination(txy, recovery_xy, return_dist),
                    payload_override=self._uav_expected_payload_after_task(str(uid), task),
                )
            )
            req_total = float(req_go + req_back + reserve + return_margin + rendez_margin)
            if batt + 1e-9 >= req_total:
                feasible += 1
        return float(np.clip(float(feasible) / max(len(uavs), 1), 0.0, 1.0))

    def _uav_cover_fraction_for_task(self, task: Optional[DeliveryTask]) -> float:
        # Backward-compatible wrapper.
        return float(self._task_uav_cover_fraction(task))

    def _is_high_pressure_emergency_task(self, task: Optional[DeliveryTask]) -> bool:
        if task is None or task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
            return False
        pending_e = int(self._pending_emergency_task_count())
        active_uav = int(max(self._active_uav_count(), 1))
        slack = int(self._task_deadline_slack_steps(task))
        is_island = bool(self._task_is_island(task))
        cover = float(self._task_uav_cover_fraction(task))
        overload = bool(pending_e >= int(max(active_uav + 1, 4)))
        urgent = bool(slack <= int(max(getattr(self.cfg, "uav_conditional_rendezvous_max_deadline_slack_steps", 18), 0)))
        low_cover = bool(cover < float(np.clip(getattr(self.cfg, "hrl_truck_emergency_force_relief_uav_cover_threshold", 0.35), 0.0, 1.0)))
        return bool(self._is_l_pressure_context() or is_island or overload or urgent or low_cover)

    def _is_high_pressure_island_task(self, task: Optional[DeliveryTask]) -> bool:
        if task is None or task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
            return False
        if not bool(self._task_is_island(task)):
            return False
        return bool(True)

    def _note_uav_task_reject(self, aid: str, task: Optional[DeliveryTask], reason: str) -> None:
        r = str(reason).strip().lower()
        tid = str(getattr(task, "task_id", "none")) if task is not None else "none"
        key = (int(self.state.step_index), str(aid), tid, r)
        if key in self._diag_uav_task_reject_seen_step:
            return
        self._diag_uav_task_reject_seen_step.add(key)
        if r in {"below_launch_min", "post_bind_recharge", "post_bind_dwell"}:
            self.uav_task_reject_below_launch_min_count += 1
        elif r in {"not_loaded", "post_bind_reload"}:
            self.uav_task_reject_not_loaded_count += 1
        elif r in {"insufficient_recovery_margin", "rendezvous_launch_disabled", "no_truck_for_return", "recovery_margin"}:
            self.uav_task_reject_recovery_margin_count += 1
        elif r in {"horizon", "horizon_insufficient"}:
            self.uav_task_reject_horizon_count += 1
        elif r in {"comm_block", "comm_degraded"}:
            self.uav_task_reject_comm_block_count += 1
        else:
            self.uav_task_reject_corridor_count += 1

    def _note_truck_emergency_relief_gate(self, aid: str, task: Optional[DeliveryTask], blocked_by_normal_guard: bool = False, override: bool = False) -> None:
        if task is None or task.kind != TaskKind.EMERGENCY:
            return
        tid = str(getattr(task, "task_id", "none"))
        key = (int(self.state.step_index), str(aid), tid)
        if key in self._diag_truck_emergency_relief_seen_step:
            return
        self._diag_truck_emergency_relief_seen_step.add(key)
        if bool(blocked_by_normal_guard):
            self.truck_emergency_blocked_by_normal_guard_count += 1
        if bool(override):
            self.truck_emergency_relief_override_count += 1

    def _note_truck_emergency_serviceability(self, aid: str, task: Optional[DeliveryTask], serviceable: bool) -> None:
        if task is None or task.kind != TaskKind.EMERGENCY:
            return
        tid = str(getattr(task, "task_id", "none"))
        key = (int(self.state.step_index), str(aid), tid)
        if key in self._diag_truck_emergency_serviceability_seen_step:
            return
        self._diag_truck_emergency_serviceability_seen_step.add(key)
        if bool(serviceable):
            self.truck_emergency_serviceable_count += 1
        else:
            self.truck_emergency_not_serviceable_count += 1

    def _is_high_pressure_recovery_corridor_feasible(
        self,
        aid: str,
        task: Optional[DeliveryTask],
        launch_reason: str = "",
    ) -> bool:
        if task is None or task.kind != TaskKind.EMERGENCY:
            return False
        s = self.state.agents.get(str(aid), None)
        if s is None or s.kind != AgentKind.UAV or bool(getattr(s, "crashed", False)):
            return False
        batt = float(max(getattr(s, "battery", 0.0), 0.0))
        cur_xy = s.pos_xy if s.pos_xy is not None else self._node_xy(int(s.node or 0))
        back_tid, d_back = self._nearest_truck_from_xy(cur_xy)
        if not np.isfinite(d_back):
            return False
        reserve = float(np.clip(getattr(self.cfg, "uav_emergency_reserve_fraction", 0.20), 0.0, 1.0))
        rendez_margin = float(np.clip(getattr(self.cfg, "uav_rendezvous_margin_fraction", 0.10), 0.0, 1.0))
        bind_window = float(max(self._uav_bind_window_m(), 1.0))
        recovery_buf_eff = float(self._effective_recovery_buffer_for_sortie(str(aid), task, launch_reason=launch_reason))
        recovery_dist = float(max(bind_window, recovery_buf_eff))
        return_dist = float(d_back + recovery_dist)
        recovery_xy = cur_xy
        if back_tid is not None and str(back_tid) in self.state.agents:
            back_state = self.state.agents[str(back_tid)]
            recovery_xy = back_state.pos_xy if back_state.pos_xy is not None else self._node_xy(int(back_state.node or 0))
        req = float(
            self._uav_energy_cost_fraction(
                str(aid), return_dist, cur_xy,
                destination=self._uav_extended_destination(cur_xy, recovery_xy, return_dist),
                payload_override=float(getattr(s, "payload_kg_current", 0.0)),
            )
        )
        return bool(batt + 1e-9 >= float(req + reserve + rendez_margin))

    def _pending_emergency_task_count(self) -> int:
        return int(
            sum(
                1
                for task in self.state.tasks.values()
                if task.kind == TaskKind.EMERGENCY and task.status in (TaskStatus.PENDING, TaskStatus.CLAIMED)
            )
        )

    def _active_uav_count(self) -> int:
        return int(
            sum(
                1
                for s in self.state.agents.values()
                if s.kind == AgentKind.UAV and (not bool(getattr(s, "crashed", False)))
            )
        )

    def _uav_launch_min_battery_threshold_for_task(self, task: Optional[DeliveryTask]) -> float:
        launch_field = "uav_min_takeoff_soc" if str(getattr(self.cfg, "physical_environment_version", "v1")).lower() == "v2" else "uav_launch_min_battery_fraction"
        base = float(np.clip(getattr(self.cfg, launch_field, 0.10), 0.0, 1.0))
        threshold = float(base)
        if bool(getattr(self.cfg, "uav_adaptive_launch_gate_enabled", True)):
            pending_e = int(self._pending_emergency_task_count())
            active_uav = int(max(self._active_uav_count(), 1))
            if pending_e >= active_uav:
                floor = float(np.clip(getattr(self.cfg, "uav_adaptive_launch_min_floor", 0.52), 0.0, 1.0))
                relax = float(np.clip(getattr(self.cfg, "uav_adaptive_launch_relax_delta", 0.06), 0.0, 1.0))
                adaptive_candidate = float(max(floor, base - relax))
                # Adaptive gate can only relax (lower), never raise threshold.
                threshold = float(min(threshold, adaptive_candidate))
        if task is not None and bool(self._is_high_pressure_emergency_task(task) or self._is_high_pressure_island_task(task)):
            hp_floor = float(np.clip(getattr(self.cfg, "uav_high_pressure_launch_min_floor", 0.48), 0.0, 1.0))
            threshold = float(min(threshold, hp_floor))
        return float(np.clip(threshold, 0.0, 1.0))

    def _uav_launch_min_battery_threshold(self, task: Optional[DeliveryTask] = None) -> float:
        return float(self._uav_launch_min_battery_threshold_for_task(task))

    def _uav_force_takeoff_battery_threshold_for_task(self, task: Optional[DeliveryTask]) -> float:
        base = float(np.clip(getattr(self.cfg, "uav_force_takeoff_battery_threshold", 0.95), 0.0, 1.0))
        launch_thr = float(self._uav_launch_min_battery_threshold_for_task(task))
        threshold = float(base)
        if bool(getattr(self.cfg, "uav_adaptive_launch_gate_enabled", True)):
            pending_e = int(self._pending_emergency_task_count())
            active_uav = int(max(self._active_uav_count(), 1))
            if pending_e >= active_uav:
                gap = float(np.clip(getattr(self.cfg, "uav_adaptive_force_takeoff_gap", 0.16), 0.0, 1.0))
                threshold = float(max(launch_thr + gap, base - 0.08))
        if task is not None and bool(self._is_high_pressure_emergency_task(task) or self._is_high_pressure_island_task(task)):
            hp_gap = float(np.clip(getattr(self.cfg, "uav_high_pressure_force_takeoff_gap", 0.12), 0.0, 1.0))
            threshold = float(min(threshold, max(launch_thr + hp_gap, base - 0.12)))
        return float(np.clip(threshold, 0.0, 1.0))

    def _uav_force_takeoff_battery_threshold(self, task: Optional[DeliveryTask] = None) -> float:
        return float(self._uav_force_takeoff_battery_threshold_for_task(task))

    def _effective_recovery_buffer_for_sortie(
        self,
        aid: str,
        task: Optional[DeliveryTask],
        launch_reason: str = "",
    ) -> float:
        base_buf = float(max(getattr(self.cfg, "uav_recovery_distance_buffer_m", 600.0), 0.0))
        if task is None or task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
            return float(base_buf)
        is_hp = bool(self._is_high_pressure_emergency_task(task) or self._is_high_pressure_island_task(task))
        if not is_hp:
            return float(base_buf)

        rs = str(launch_reason)
        hp_bonus = float(max(getattr(self.cfg, "uav_high_pressure_recovery_margin_bonus_m", 250.0), 0.0))
        relaxed_bonus = float(max(getattr(self.cfg, "uav_relaxed_rendezvous_recovery_margin_bonus_m", 300.0), 0.0))
        if rs.startswith("direct_safe"):
            return float(max(base_buf - 0.5 * hp_bonus, 0.0))
        if rs.startswith("rendezvous_safe_relaxed"):
            d_nearest_truck = float(self._task_nearest_truck_distance_m(task))
            hp_nearest_cap = float(max(getattr(self.cfg, "uav_high_pressure_rendezvous_max_nearest_truck_m", 2200.0), 0.0))
            if np.isfinite(d_nearest_truck) and d_nearest_truck <= hp_nearest_cap:
                return float(max(base_buf - max(hp_bonus, relaxed_bonus), 0.0))
        if rs.startswith("rendezvous_safe"):
            return float(max(base_buf - hp_bonus, 0.0))
        return float(base_buf)

    def _record_uav_launch_gate_result(self, ok: bool, reason: str) -> None:
        self.uav_launch_gate_enter_count += 1
        r = str(reason).strip().lower()
        if bool(ok):
            if r == "direct_safe":
                self.uav_launch_gate_direct_safe_count += 1
            elif r.startswith("rendezvous_safe_relaxed"):
                self.uav_launch_gate_rendezvous_safe_relaxed_count += 1
            elif r.startswith("rendezvous_safe"):
                self.uav_launch_gate_rendezvous_safe_count += 1
            else:
                self.uav_launch_gate_block_other_count += 1
            return
        if r == "below_launch_min":
            self.uav_launch_gate_block_below_launch_min_count += 1
        elif r in {"insufficient_recovery_margin", "rendezvous_launch_disabled"}:
            self.uav_launch_gate_block_recovery_margin_count += 1
        elif r in {"no_truck_for_return", "corridor_blocked"}:
            self.uav_launch_gate_block_corridor_count += 1
        else:
            self.uav_launch_gate_block_other_count += 1

    def _uav_launch_block_cooldown_active(self, aid: str) -> bool:
        until = int(self._uav_launch_block_cooldown_until_step.get(str(aid), -1))
        return bool(int(self.state.step_index) <= until)

    def _record_uav_launch(self, aid: str) -> None:
        s = self.state.agents.get(str(aid), None)
        if s is None or s.kind != AgentKind.UAV:
            return
        batt = float(np.clip(float(getattr(s, "battery", 0.0)), 0.0, 1.0))
        self.uav_launch_count_total += 1
        self.uav_launch_battery_fraction_sum += float(batt)
        if int(self.uav_launch_count_total) <= 1:
            self.uav_launch_battery_fraction_min = float(batt)
        else:
            self.uav_launch_battery_fraction_min = float(min(float(self.uav_launch_battery_fraction_min), float(batt)))
        # A launch is only legal for a loaded emergency-delivery task.  Record
        # that task as an environment-side contract so later replans cannot
        # turn the sortie into an empty or retargeted flight.
        if bool(getattr(self.cfg, "uav_strict_sortie_contract_enabled", True)) and bool(self._uav_loaded(str(aid))):
            gid = self._effective_goals.get(str(aid), self._recommended_goals.get(str(aid), None))
            task = self.state.tasks.get(str(gid)) if gid is not None else None
            if task is not None and task.kind == TaskKind.EMERGENCY and task.status == TaskStatus.PENDING:
                self._uav_sortie_contract_task[str(aid)] = str(task.task_id)
                self._uav_sortie_contract_version[str(aid)] = int(
                    max(getattr(task, "route_contract_version", 0), 0)
                )

    def _commit_uav_delivery_launch(
        self,
        aid: str,
        task: Optional[DeliveryTask],
        launch_reason: str,
        launch_truck_id: str = "",
    ) -> bool:
        """Apply the one authoritative state transition for a delivery launch."""
        uid = str(aid)
        state = self.state.agents.get(uid, None)
        if (
            state is None
            or state.kind != AgentKind.UAV
            or task is None
            or not self._task_is_uav_delivery(task)
            or task.status != TaskStatus.PENDING
            or not bool(self._uav_loaded_for_task(uid, task))
        ):
            return False

        reason = str(launch_reason)
        state.follow_target = None
        state.replenish_timer = 0
        state.uav_reload_timer = 0
        self._uav_last_launch_reason[uid] = reason
        self._uav_sortie_relaxed_latch[uid] = bool(
            reason.startswith("rendezvous_safe_relaxed")
        )
        self.uav_safe_launch_count_total += 1
        if reason == "direct_safe":
            self.uav_launch_direct_safe_count += 1
        elif reason.startswith("rendezvous_safe_relaxed"):
            self.uav_launch_rendezvous_safe_relaxed_count += 1
            self.relaxed_sortie_selected_count_total += 1
        elif reason.startswith("rendezvous_safe"):
            self.uav_launch_rendezvous_safe_count += 1
        self._record_uav_launch(uid)
        if (
            reason.startswith("rendezvous_safe")
            and bool(
                getattr(
                    self.cfg,
                    "uav_rendezvous_planned_recovery_request_enabled",
                    True,
                )
            )
            and str(launch_truck_id)
        ):
            self._uav_recovery_requested_truck[uid] = str(launch_truck_id)
        self._uav_bind_commit_target[uid] = None
        self._uav_bind_commit_until_step[uid] = -1
        return True

    def _note_unsafe_launch_attempt(self, aid: str, reason: str = "") -> None:
        self.uav_unsafe_launch_attempt_count_total += 1
        self.uav_unsafe_launch_block_count_total += 1
        self.uav_launch_block_unsafe_count += 1
        cooldown = int(max(getattr(self.cfg, "uav_unsafe_launch_block_cooldown_steps", 0), 0))
        if cooldown > 0:
            self._uav_launch_block_cooldown_until_step[str(aid)] = int(self.state.step_index + cooldown)
        s = self.state.agents.get(str(aid), None)
        batt = float(getattr(s, "battery", 0.0)) if s is not None else 0.0
        gid = self._effective_goals.get(str(aid), self._recommended_goals.get(str(aid), None))
        task = self.state.tasks.get(str(gid)) if gid is not None else None
        if batt + 1e-9 < self._uav_launch_min_battery_threshold(task=task) or str(reason) == "below_launch_min":
            self.uav_low_battery_illegal_launch_count_total += 1

    def _uav_recovery_required(self, aid: str) -> bool:
        s = self.state.agents[str(aid)]
        if s.kind != AgentKind.UAV or bool(getattr(s, "crashed", False)):
            return False

        batt = float(max(getattr(s, "battery", 0.0), 0.0))
        force_thr = float(np.clip(getattr(self.cfg, "uav_low_battery_force_recover_threshold", 0.25), 0.0, 1.0))
        if batt <= force_thr:
            return True

        if bool(getattr(s, "uav_needs_reload_flag", False)) or (not bool(self._uav_loaded(str(aid)))):
            return True

        # In-flight weather above the recovery threshold takes precedence over
        # the accepted-sortie delivery bypass.  The aircraft must enter the
        # explicit recovery state rather than continue an ordinary task.
        if s.follow_target is None:
            cur_xy_weather = s.pos_xy if s.pos_xy is not None else self._node_xy(int(s.node or 0))
            if self._uav_weather_safety_reason(cur_xy_weather, recovery=True):
                return True

        # A forced-RTH latch is allowed to coexist with an accepted loaded
        # sortie while the remaining delivery leg is still energy-feasible.
        # The post-delivery recovery path remains authoritative once the task
        # is completed or the outbound leg loses feasibility.
        if (
            bool(self._uav_forced_rth_latch.get(str(aid), False))
            and self._uav_sortie_delivery_recovery_bypass_active(str(aid))
        ):
            return False

        # Distance-aware hard recovery gate:
        # even above force threshold, trigger recovery early when remaining battery
        # cannot safely cover conservative rendezvous distance to nearest truck.
        cur_xy = s.pos_xy if s.pos_xy is not None else self._node_xy(int(s.node or 0))
        back_tid, d_back = self._nearest_truck_from_xy(cur_xy)
        if np.isfinite(d_back):
            reserve = float(np.clip(getattr(self.cfg, "uav_emergency_reserve_fraction", 0.20), 0.0, 1.0))
            return_margin = float(np.clip(getattr(self.cfg, "uav_return_margin_fraction", 0.15), 0.0, 1.0))
            gid = self._effective_goals.get(str(aid), self._recommended_goals.get(str(aid), None))
            goal_task = self.state.tasks.get(str(gid)) if gid is not None else None
            launch_reason = str(self._uav_last_launch_reason.get(str(aid), ""))
            recovery_buf_eff = float(self._effective_recovery_buffer_for_sortie(str(aid), goal_task, launch_reason=launch_reason))
            if (
                goal_task is not None
                and goal_task.kind == TaskKind.EMERGENCY
                and goal_task.status == TaskStatus.PENDING
                and str(launch_reason).startswith("rendezvous_safe")
                and bool(self._is_high_pressure_recovery_corridor_feasible(str(aid), goal_task, launch_reason=launch_reason))
            ):
                bind_window = float(max(self._uav_bind_window_m(), 0.0))
                recovery_buf_eff = float(max(recovery_buf_eff, bind_window))

            req_back = float(
                self._uav_energy_cost_fraction(
                    str(aid),
                    float(d_back + recovery_buf_eff),
                    (float(cur_xy[0]), float(cur_xy[1])),
                    destination=self._uav_extended_destination(
                        cur_xy,
                        self._agent_xy(str(back_tid)) if back_tid is not None else cur_xy,
                        float(d_back + recovery_buf_eff),
                    ),
                    payload_override=float(getattr(s, "payload_kg_current", 0.0)),
                )
            )
            if batt + 1e-9 < float(req_back + reserve + return_margin):
                return True

        return bool(self._uav_forced_rth_latch.get(str(aid), False))

    def _uav_sortie_delivery_leg_feasible(
        self, aid: str, task: Optional[DeliveryTask] = None
    ) -> bool:
        """Whether an installed loaded sortie can still reach its task.

        This is deliberately an outbound-only check.  Recovery feasibility is
        evaluated after delivery; using a complete round-trip requirement here
        would let a forced-RTH latch strand an otherwise deliverable sortie.
        """
        uid = str(aid)
        s = self.state.agents.get(uid, None)
        if s is None or s.kind != AgentKind.UAV or bool(getattr(s, "crashed", False)):
            return False
        contract_tid = self._uav_sortie_contract_task.get(uid, None)
        if task is None:
            task = self.state.tasks.get(str(contract_tid)) if contract_tid is not None else None
        elif contract_tid is None or str(getattr(task, "task_id", "")) != str(contract_tid):
            return False
        if (
            task is None
            or not self._task_is_uav_delivery(task)
            or task.status not in (TaskStatus.PENDING, TaskStatus.CLAIMED)
            or not bool(self._uav_loaded_for_task(uid, task))
        ):
            return False
        try:
            cur_xy = self._agent_xy(uid)
            delivery_dist = float(self._agent_distance_to_task(uid, task))
            delivery_energy = float(
                self._uav_energy_cost_fraction(
                    uid,
                    delivery_dist,
                    cur_xy,
                    destination=self._node_xy(int(task.demand_node)),
                    payload_override=float(getattr(s, "payload_kg_current", 0.0)),
                )
            )
            service_buffer = float(
                max(getattr(self.cfg, "service_battery_buffer", 0.0), 0.0)
            )
        except Exception:
            return False
        return bool(
            np.isfinite(delivery_dist)
            and np.isfinite(delivery_energy)
            and delivery_dist >= 0.0
            and delivery_energy >= 0.0
            and float(getattr(s, "battery", 0.0)) + 1e-9
            >= delivery_energy + service_buffer
        )

    def _uav_sortie_delivery_recovery_bypass_active(
        self, aid: str, task: Optional[DeliveryTask] = None
    ) -> bool:
        """Honor any accepted loaded delivery leg that remains feasible.

        Direct-safe and rendezvous-safe launches share the same physical
        contract once airborne.  A forced-RTH latch may interrupt only when
        the outbound delivery leg itself no longer fits the remaining energy.
        """
        return bool(
            _UAV_SORTIE_DELIVERY_LEG_RECOVERY_BYPASS_ENABLED
            and self._uav_sortie_delivery_leg_feasible(str(aid), task)
        )

    def _uav_hard_recovery_required(self, aid: str) -> bool:
        """Single hard-safety definition shared by UAV and truck control."""
        uid = str(aid)
        state = self.state.agents.get(uid, None)
        if (
            state is None
            or state.kind != AgentKind.UAV
            or bool(getattr(state, "crashed", False))
            or state.follow_target is not None
        ):
            return False
        force_thr = float(
            np.clip(
                getattr(self.cfg, "uav_low_battery_force_recover_threshold", 0.25),
                0.0,
                1.0,
            )
        )
        if float(getattr(state, "battery", 0.0)) <= force_thr:
            return True
        if bool(self._uav_forced_rth_latch.get(uid, False)):
            return not bool(self._uav_sortie_delivery_recovery_bypass_active(uid))
        low_thr = float(
            np.clip(
                getattr(self.cfg, "uav_low_battery_goal_lock_threshold", 0.35),
                0.0,
                1.0,
            )
        )
        needs_reload = bool(getattr(state, "uav_needs_reload_flag", False)) or (
            not bool(self._uav_loaded(uid))
        )
        return bool(needs_reload and float(getattr(state, "battery", 0.0)) <= low_thr)

    def _has_hard_recovery_uav(self) -> bool:
        for uid, us in self.state.agents.items():
            if us.kind == AgentKind.UAV and self._uav_hard_recovery_required(str(uid)):
                return True
        return False

    def _has_airborne_hard_recovery_uav(self) -> bool:
        """True only for hard UAV safety cases that can interrupt truck final approach."""
        for uid, us in self.state.agents.items():
            if (
                us.kind == AgentKind.UAV
                and getattr(us, "pos_xy", None) is not None
                and self._uav_hard_recovery_required(str(uid))
            ):
                return True
        return False

    def _truck_has_assigned_airborne_hard_recovery_request(self, aid: str) -> bool:
        """Whether this truck is the explicit recovery anchor for a hard airborne UAV."""
        req_map = getattr(self, "_uav_recovery_requested_truck", {}) if hasattr(self, "_uav_recovery_requested_truck") else {}
        for uid, target_truck in dict(req_map).items():
            if str(target_truck) != str(aid):
                continue
            us = self.state.agents.get(str(uid), None)
            if us is None or us.kind != AgentKind.UAV or bool(getattr(us, "crashed", False)):
                continue
            if us.follow_target is not None or getattr(us, "pos_xy", None) is None:
                continue
            if self._uav_hard_recovery_required(str(uid)):
                return True
        return False

    def _truck_active_multiround_routine_commitment(self, aid: str) -> Optional[str]:
        """Return a live partial-service commitment for this truck, if any."""
        if not bool(
            getattr(
                self.cfg,
                "hrl_route_plan_routine_multiround_commitment_enabled",
                True,
            )
        ):
            return None
        task_id = dict(
            getattr(self, "_routine_multiround_service_commitment_by_truck", {})
        ).get(str(aid), None)
        task = self.state.tasks.get(str(task_id), None) if task_id is not None else None
        if (
            task is None
            or task.kind != TaskKind.NORMAL
            or task.status not in (TaskStatus.PENDING, TaskStatus.CLAIMED)
            or float(getattr(task, "remaining_demand_kg", 0.0)) <= 1e-9
        ):
            commitments = getattr(
                self, "_routine_multiround_service_commitment_by_truck", None
            )
            if commitments is None:
                commitments = {}
                self._routine_multiround_service_commitment_by_truck = commitments
            commitments.pop(str(aid), None)
            return None
        state = self.state.agents.get(str(aid), None)
        if (
            state is None
            or state.kind != AgentKind.TRUCK
            or state.node is None
            or int(state.node) != int(task.demand_node)
            or self._truck_requires_depot(str(aid))
            or not self._truck_can_service_task(str(aid), task)
        ):
            return None
        return str(task.task_id)

    def _truck_routine_goal_support_protected(self, aid: str, neighbors: Optional[List[int]] = None) -> bool:
        """Keep a truck on the last short leg of a routine task.

        The check uses a short step horizon, but also looks one legal truck move
        ahead. That protects cases where the truck is one intersection away from
        a decisive final approach, without broadly disabling UAV support.
        """
        if not bool(getattr(self.cfg, "truck_routine_near_goal_support_protect_enabled", True)):
            return False
        s = self.state.agents.get(str(aid), None)
        if s is None or s.kind != AgentKind.TRUCK or bool(getattr(s, "crashed", False)):
            return False
        if s.node is None or self._truck_requires_depot(str(aid)):
            return False
        multiround_task_id = self._truck_active_multiround_routine_commitment(str(aid))
        goal_id = (
            multiround_task_id
            if multiround_task_id is not None
            else self._effective_goals.get(
                str(aid), self._recommended_goals.get(str(aid), None)
            )
        )
        task = self.state.tasks.get(str(goal_id), None) if goal_id is not None else None
        if task is None or task.kind != TaskKind.NORMAL or task.status != TaskStatus.PENDING:
            return False
        if not self._truck_can_service_task(str(aid), task):
            return False

        steps = int(
            max(
                getattr(
                    self.cfg,
                    "hrl_routine_near_completion_eta_steps",
                    getattr(self.cfg, "truck_routine_near_goal_support_protect_steps", 5),
                ),
                0,
            )
        )
        if steps <= 0:
            return False
        speed = float(max(getattr(self.cfg, "truck_speed_mps", 0.0), 0.0))
        dt_s = float(max(getattr(self.cfg, "dt_seconds", getattr(self.cfg, "dt", 1.0)), 1e-6))
        protect_dist = float(max(speed * dt_s * float(steps), float(getattr(self.cfg, "hrl_routine_near_completion_route_dist_m", 0.0)), 1.0))
        target_node = int(task.demand_node)

        try:
            cur_dist = float(self._decision_shortest_path_distance(int(s.node), target_node))
        except Exception:
            cur_dist = float("inf")
        if np.isfinite(cur_dist) and cur_dist <= protect_dist:
            return True

        nbs = neighbors
        if nbs is None:
            try:
                nbs = list(self._decision_neighbors(int(s.node)))
            except Exception:
                nbs = []
        for nb in nbs or []:
            try:
                nb_dist = float(self._decision_shortest_path_distance(int(nb), target_node))
            except Exception:
                nb_dist = float("inf")
            if np.isfinite(nb_dist) and nb_dist <= protect_dist:
                return True
        return False

    def _routine_protection_tc_override_allowed(
        self,
        aid: str,
        support_nb: Optional[int],
    ) -> Tuple[bool, Dict[str, float]]:
        """Permit a short TC support move through routine protection only with evidence.

        This is not a new support-chain mechanism; it only decides whether an
        already proposed support/recovery neighbor may temporarily override the
        routine near-completion guard.
        """
        info: Dict[str, float] = {
            "uav_id": "",
            "task_id": "",
            "launch_gain_m": 0.0,
            "routine_delay_steps": float("inf"),
            "launchable": 0.0,
            "recovery_feasible": 0.0,
            "reject_reason": "",
        }
        if support_nb is None or not bool(getattr(self.cfg, "hrl_routine_protection_tc_override_enabled", True)):
            return False, info
        truck = self.state.agents.get(str(aid), None)
        if truck is None or truck.kind != AgentKind.TRUCK or truck.node is None:
            return False, info
        goal_id = self._effective_goals.get(str(aid), self._recommended_goals.get(str(aid), None))
        routine_task = self.state.tasks.get(str(goal_id), None) if goal_id is not None else None
        if routine_task is None or routine_task.kind != TaskKind.NORMAL or routine_task.status != TaskStatus.PENDING:
            return False, info

        speed = float(max(getattr(self.cfg, "truck_speed_mps", 0.0), 1e-6))
        dt_s = float(max(getattr(self.cfg, "dt_seconds", getattr(self.cfg, "dt", 1.0)), 1e-6))
        try:
            cur_routine_d = float(self._decision_shortest_path_distance(int(truck.node), int(routine_task.demand_node)))
            nb_routine_d = float(self._decision_shortest_path_distance(int(support_nb), int(routine_task.demand_node)))
            move_d = float(self.topology.edge_distance(int(truck.node), int(support_nb)))
        except Exception:
            return False, info
        if not (np.isfinite(cur_routine_d) and np.isfinite(nb_routine_d) and np.isfinite(move_d)):
            return False, info
        routine_delay_steps = float(max((move_d + nb_routine_d - cur_routine_d) / max(speed * dt_s, 1e-6), 0.0))
        info["routine_delay_steps"] = float(routine_delay_steps)
        if routine_delay_steps > float(max(getattr(self.cfg, "hrl_routine_protection_tc_override_max_routine_delay_steps", 3), 0)):
            self.routine_near_completion_tc_override_reject_delay_count += 1
            info["reject_reason"] = "routine_delay"
            return False, info

        cur_xy = self._node_xy(int(truck.node))
        nb_xy = self._node_xy(int(support_nb))
        best: Tuple[float, Optional[str], Optional[DeliveryTask], bool] = (-1e18, None, None, False)
        require_loaded = bool(getattr(self.cfg, "hrl_routine_protection_tc_override_require_loaded_uav", True))
        require_recovery = bool(getattr(self.cfg, "hrl_routine_protection_tc_override_require_recovery_feasible", True))
        min_gain = float(max(getattr(self.cfg, "hrl_routine_protection_tc_override_min_launch_gain_m", 300.0), 0.0))
        max_support_steps = int(max(getattr(self.cfg, "hrl_routine_protection_tc_override_max_support_steps", 5), 0))
        support_dist_cap = float(max_support_steps) * speed * dt_s
        loaded_uav_seen = False
        for uid, us in self.state.agents.items():
            if us.kind != AgentKind.UAV or bool(getattr(us, "crashed", False)):
                continue
            if us.follow_target is None or str(us.follow_target) != str(aid):
                continue
            if require_loaded and (bool(getattr(us, "uav_needs_reload_flag", False)) or not bool(self._uav_loaded(str(uid)))):
                continue
            loaded_uav_seen = True
            # A TC override should not require the UAV to already have a stable
            # emergency goal. The whole point is to let a short support move
            # create a launch opportunity for a loaded docked UAV.
            ugoal = self._effective_goals.get(str(uid), self._recommended_goals.get(str(uid), None))
            candidate_tasks: List[DeliveryTask] = []
            cur_task = self.state.tasks.get(str(ugoal), None) if ugoal is not None else None
            if cur_task is not None and cur_task.kind == TaskKind.EMERGENCY:
                candidate_tasks.append(cur_task)
            for t in self.state.tasks.values():
                if t.kind == TaskKind.EMERGENCY and str(t.task_id) != str(getattr(cur_task, "task_id", "")):
                    candidate_tasks.append(t)

            for task in candidate_tasks:
                if task is None or task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
                    continue
                if task.in_service_by is not None:
                    continue
                tnode = self.topology.nodes[int(task.demand_node)]
                tx, ty = float(tnode.x), float(tnode.y)
                cur_d = float(np.hypot(float(cur_xy[0]) - tx, float(cur_xy[1]) - ty))
                nb_d = float(np.hypot(float(nb_xy[0]) - tx, float(nb_xy[1]) - ty))
                gain = float(cur_d - nb_d)
                launchable = bool(self._uav_docked_task_actionable_now(str(uid), task))
                near_launchable = bool(gain >= min_gain and move_d <= max(support_dist_cap, 1.0))
                recovery_feasible = bool(launchable or near_launchable)
                if require_recovery and not recovery_feasible:
                    self.routine_near_completion_tc_override_reject_recovery_count += 1
                    info["reject_reason"] = "recovery"
                    continue
                if not (launchable or near_launchable):
                    self.routine_near_completion_tc_override_reject_not_near_launchable_count += 1
                    info["reject_reason"] = "not_near_launchable"
                    continue
                self.tc_override_candidate_count += 1
                full_ok = True
                block_reason = "ok"
                full_diag: Dict[str, float] = {}
                if bool(getattr(self.cfg, "hrl_routine_protection_delivery_feasible_tc_override_enabled", True)):
                    full_ok, block_reason, full_diag = self._is_tc_override_delivery_feasible(
                        str(aid),
                        str(uid),
                        task,
                        int(support_nb),
                        routine_delay_steps=float(routine_delay_steps),
                        launch_gain_m=float(gain),
                        launchable_now=bool(launchable),
                    )
                    if full_diag.get("predicted_launchable", 0.0) > 0.0:
                        self.tc_override_predicted_launchable_count += 1
                    if full_diag.get("predicted_full_sortie_feasible", 0.0) > 0.0:
                        self.tc_override_predicted_delivery_feasible_count += 1
                    if not full_ok:
                        self.tc_override_blocked_not_full_sortie_feasible_count += 1
                        if block_reason == "low_recovery_margin":
                            self.tc_override_blocked_low_recovery_margin_count += 1
                        elif block_reason == "low_battery_margin":
                            self.tc_override_blocked_low_battery_margin_count += 1
                        elif block_reason == "recent_reject":
                            self.tc_override_blocked_recent_reject_count += 1
                        elif block_reason == "lifeline_risk":
                            self.tc_override_blocked_lifeline_risk_count += 1
                        elif block_reason == "routine_delay":
                            self.tc_override_blocked_routine_delay_count += 1
                        self._tc_override_trace_rows.append({
                            "step": int(self.state.step_index),
                            "truck_id": str(aid),
                            "routine_task_id": str(routine_task.task_id),
                            "route_dist_to_routine": float(cur_routine_d),
                            "eta_to_routine": float(cur_routine_d / max(speed * dt_s, 1e-6)),
                            "uav_id": str(uid),
                            "tc_task_id": str(task.task_id),
                            "launch_gain_m": float(gain),
                            "routine_delay_steps": float(routine_delay_steps),
                            "predicted_launchable": int(full_diag.get("predicted_launchable", 0.0) > 0.0),
                            "predicted_full_sortie_feasible": int(full_diag.get("predicted_full_sortie_feasible", 0.0) > 0.0),
                            "predicted_recovery_margin_m": float(full_diag.get("predicted_recovery_margin_m", float("nan"))),
                            "predicted_battery_margin_ratio": float(full_diag.get("predicted_battery_margin_ratio", float("nan"))),
                            "predicted_lifeline_remaining": float(full_diag.get("predicted_lifeline_remaining", 0.0)),
                            "recent_reject_hit": int(full_diag.get("recent_reject_hit", 0.0) > 0.0),
                            "override_allowed": 0,
                            "override_block_reason": str(block_reason),
                            "actual_launch_after": 0,
                            "actual_delivery_after": 0,
                            "actual_forced_recovery_after": 0,
                            "actual_reject_reason_after": "",
                        })
                        continue
                score = float(gain + (500.0 if launchable else 0.0))
                if score > best[0]:
                    best = (score, str(uid), task, bool(launchable))
                    info.update({
                        "uav_id": str(uid),
                        "task_id": str(task.task_id),
                        "launch_gain_m": float(gain),
                        "launchable": 1.0 if launchable else 0.0,
                        "recovery_feasible": 1.0 if recovery_feasible else 0.0,
                        "predicted_full_sortie_feasible": 1.0 if full_ok else 0.0,
                    })
                    self._tc_override_trace_rows.append({
                        "step": int(self.state.step_index),
                        "truck_id": str(aid),
                        "routine_task_id": str(routine_task.task_id),
                        "route_dist_to_routine": float(cur_routine_d),
                        "eta_to_routine": float(cur_routine_d / max(speed * dt_s, 1e-6)),
                        "uav_id": str(uid),
                        "tc_task_id": str(task.task_id),
                        "launch_gain_m": float(gain),
                        "routine_delay_steps": float(routine_delay_steps),
                        "predicted_launchable": int(bool(launchable) or float(full_diag.get("predicted_launchable", 0.0)) > 0.0),
                        "predicted_full_sortie_feasible": int(bool(full_ok)),
                        "predicted_recovery_margin_m": float(full_diag.get("predicted_recovery_margin_m", float("nan"))),
                        "predicted_battery_margin_ratio": float(full_diag.get("predicted_battery_margin_ratio", float("nan"))),
                        "predicted_lifeline_remaining": float(full_diag.get("predicted_lifeline_remaining", 0.0)),
                        "recent_reject_hit": int(full_diag.get("recent_reject_hit", 0.0) > 0.0),
                        "override_allowed": 1,
                        "override_block_reason": "",
                        "actual_launch_after": 0,
                        "actual_delivery_after": 0,
                        "actual_forced_recovery_after": 0,
                        "actual_reject_reason_after": "",
                    })
        if best[1] is None:
            if not loaded_uav_seen:
                self.routine_near_completion_tc_override_reject_no_loaded_uav_count += 1
                info["reject_reason"] = "no_loaded_uav"
            elif not info.get("reject_reason"):
                self.routine_near_completion_tc_override_reject_no_candidate_count += 1
                info["reject_reason"] = "no_candidate"
            return False, info
        return True, info

    def _is_tc_override_delivery_feasible(
        self,
        truck_id: str,
        uav_id: str,
        task: DeliveryTask,
        support_node: int,
        routine_delay_steps: float,
        launch_gain_m: float,
        launchable_now: bool,
    ) -> Tuple[bool, str, Dict[str, float]]:
        """Predict whether a TC override can close launch -> delivery -> recovery.

        This intentionally mirrors the core launch/recovery constraints rather
        than using launch distance alone. It is conservative: if prediction is
        uncertain, routine protection wins.
        """
        diag: Dict[str, float] = {
            "predicted_launchable": 0.0,
            "predicted_full_sortie_feasible": 0.0,
            "predicted_recovery_margin_m": float("-inf"),
            "predicted_battery_margin_ratio": float("-inf"),
            "predicted_lifeline_remaining": 0.0,
            "recent_reject_hit": 0.0,
            "launch_gain_m": float(launch_gain_m),
            "routine_delay_steps": float(routine_delay_steps),
        }
        if task is None or task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
            return False, "task_not_pending_emergency", diag
        if task.in_service_by is not None:
            return False, "task_already_servicing", diag
        for _uid, _us in self.state.agents.items():
            if _us.kind == AgentKind.UAV and str(self._effective_goals.get(str(_uid), "")) == str(task.task_id):
                if _us.follow_target is None and getattr(_us, "pos_xy", None) is not None:
                    return False, "task_has_airborne_uav", diag
        us = self.state.agents.get(str(uav_id), None)
        if us is None or us.kind != AgentKind.UAV or bool(getattr(us, "crashed", False)):
            return False, "uav_invalid", diag
        if us.follow_target is None or str(us.follow_target) != str(truck_id):
            return False, "uav_not_on_truck", diag
        if bool(getattr(us, "uav_needs_reload_flag", False)) or (not bool(self._uav_loaded(str(uav_id)))):
            return False, "uav_not_loaded", diag
        if getattr(us, "pos_xy", None) is not None and us.follow_target is None:
            return False, "uav_airborne", diag

        max_delay = float(max(getattr(self.cfg, "hrl_tc_override_max_routine_delay_steps", 3), 0))
        if routine_delay_steps > max_delay:
            return False, "routine_delay", diag

        if bool(getattr(self.cfg, "hrl_tc_override_block_if_recent_reject", True)):
            ttl = int(max(getattr(self.cfg, "hrl_tc_override_reject_cache_ttl_steps", 20), 0))
            until = int(self._tc_override_recent_reject.get((str(uav_id), str(task.task_id)), -1))
            if ttl > 0 and until >= int(self.state.step_index):
                diag["recent_reject_hit"] = 1.0
                return False, "recent_reject", diag

        support_xy = self._node_xy(int(support_node))
        tnode = self.topology.nodes[int(task.demand_node)]
        task_xy = (float(tnode.x), float(tnode.y))
        d_go = float(np.hypot(float(support_xy[0]) - task_xy[0], float(support_xy[1]) - task_xy[1]))
        nearest_tid, d_nearest_truck = self._nearest_truck_from_xy(task_xy)
        d_back_anchor = float(np.hypot(float(support_xy[0]) - task_xy[0], float(support_xy[1]) - task_xy[1]))
        d_back = float(min(d_back_anchor, d_nearest_truck if np.isfinite(d_nearest_truck) else d_back_anchor))
        if not (np.isfinite(d_go) and np.isfinite(d_back)):
            return False, "no_recovery", diag

        reserve = float(np.clip(getattr(self.cfg, "uav_emergency_reserve_fraction", 0.20), 0.0, 1.0))
        return_margin = float(np.clip(getattr(self.cfg, "uav_return_margin_fraction", 0.15), 0.0, 1.0))
        rendez_margin = float(np.clip(getattr(self.cfg, "uav_rendezvous_margin_fraction", 0.10), 0.0, 1.0))
        recovery_buf = float(max(getattr(self.cfg, "uav_recovery_distance_buffer_m", 600.0), 0.0))
        truck_speed = float(max(getattr(self.cfg, "truck_speed_mps", 0.0), 0.0))
        decision_interval = int(max(getattr(self.cfg, "decision_interval", 5), 1))
        drift_raw = float(truck_speed * self._dt_seconds * decision_interval)
        drift_scale = float(max(getattr(self.cfg, "uav_recovery_truck_drift_margin_scale", 0.5), 0.0))
        drift_cap = float(max(getattr(self.cfg, "uav_recovery_truck_drift_margin_max_m", 600.0), 0.0))
        truck_drift_margin_m = float(min(drift_raw * drift_scale, drift_cap))
        recovery_dist = float(d_back + recovery_buf + truck_drift_margin_m)
        recovery_xy = support_xy
        if np.isfinite(d_nearest_truck) and d_nearest_truck <= d_back_anchor and nearest_tid is not None:
            nearest_state = self.state.agents[str(nearest_tid)]
            recovery_xy = nearest_state.pos_xy if nearest_state.pos_xy is not None else self._node_xy(int(nearest_state.node or 0))
        req_go = float(
            self._uav_energy_cost_fraction(
                str(uav_id), d_go, support_xy, destination=task_xy,
                payload_override=float(getattr(us, "payload_kg_current", 0.0)),
            )
        )
        req_back = float(
            self._uav_energy_cost_fraction(
                str(uav_id), recovery_dist, task_xy,
                destination=self._uav_extended_destination(task_xy, recovery_xy, recovery_dist),
                payload_override=self._uav_expected_payload_after_task(str(uav_id), task),
            )
        )
        required = float(req_go + req_back + reserve + return_margin + 0.5 * rendez_margin)
        batt = float(max(getattr(us, "battery", 0.0), 0.0))
        batt_margin = float(batt - required)
        diag["predicted_battery_margin_ratio"] = batt_margin

        if self._legacy_sortie_cap_enabled():
            sortie_cap = float(max(getattr(self.cfg, "uav_max_sortie_m", 6000.0), 1.0))
            recovery_margin_m = float(sortie_cap - (d_go + recovery_dist))
            diag["predicted_recovery_margin_m"] = recovery_margin_m
            min_rec_margin = float(max(getattr(self.cfg, "hrl_tc_override_min_recovery_margin_m", 300.0), 0.0))
            if recovery_margin_m < min_rec_margin:
                return False, "low_recovery_margin", diag
        else:
            diag["predicted_recovery_margin_m"] = float("inf")
        min_batt_margin = float(max(getattr(self.cfg, "hrl_tc_override_min_battery_margin_ratio", 0.12), 0.0))
        if batt_margin < min_batt_margin:
            return False, "low_battery_margin", diag

        # Launch can be directly safe now, or near-launchable from the support node.
        predicted_launchable = bool(launchable_now or batt + 1e-9 >= required)
        diag["predicted_launchable"] = 1.0 if predicted_launchable else 0.0
        if not predicted_launchable:
            return False, "not_launchable", diag

        uav_speed = float(max(getattr(self.cfg, "uav_max_speed_mps", 0.0), 1e-6))
        service_steps = int(max(getattr(self.cfg, "uav_service_time_steps", getattr(self.cfg, "service_time_steps", 1)), 1))
        eta_steps = int(np.ceil(d_go / max(uav_speed * self._dt_seconds, 1e-6))) + service_steps
        deadline = int(getattr(task, "deadline_step", self.cfg.max_steps))
        remaining = int(max(deadline - int(self.state.step_index), 0))
        diag["predicted_lifeline_remaining"] = float(max(remaining - eta_steps, 0))
        max_decay = float(np.clip(getattr(self.cfg, "hrl_tc_override_max_expected_lifeline_decay_ratio", 0.85), 0.0, 1.0))
        if remaining <= 0 or eta_steps > max(1, int(max_decay * float(max(remaining, 1)))):
            return False, "lifeline_risk", diag

        # A small score gate prevents launch-only improvements from overriding a
        # nearly completed routine task.
        score_gain = float(max(launch_gain_m, 0.0) / max(d_go + d_back + 1.0, 1.0))
        if score_gain < float(max(getattr(self.cfg, "hrl_tc_override_min_delivery_score_gain", 0.10), 0.0)):
            return False, "low_delivery_score_gain", diag

        diag["predicted_full_sortie_feasible"] = 1.0
        return True, "ok", diag

    def _uav_docked_task_actionable_now(self, aid: str, task: DeliveryTask) -> bool:
        s = self.state.agents.get(str(aid), None)
        if s is None or s.kind != AgentKind.UAV or bool(getattr(s, "crashed", False)):
            return False
        if task is None or not self._task_is_uav_delivery(task) or task.status != TaskStatus.PENDING:
            return False
        if s.follow_target is None:
            return bool(self.is_task_serviceable_by_agent(str(aid), task))
        if bool(getattr(s, "uav_needs_reload_flag", False)):
            return False
        if not bool(self._uav_loaded_for_task(str(aid), task)):
            return False
        ok, reason, _ = self._uav_launch_gate_check(str(aid), task=task, count_reject=False)
        if not bool(ok):
            return False
        dist_m = float(self._agent_distance_to_task(str(aid), task))
        if not np.isfinite(dist_m):
            return False
        short_cap = float(max(getattr(self.cfg, "uav_short_sortie_max_distance_m", 1200.0), 1.0))
        long_cap = float(max(short_cap * 1.60, short_cap))
        is_hp = False
        fn_hp = getattr(self, "_is_high_pressure_emergency_task", None)
        if callable(fn_hp):
            try:
                is_hp = bool(fn_hp(task))
            except Exception:
                is_hp = False
        is_island = False
        fn_island = getattr(self, "_current_island_emergency_task_ids", None)
        if callable(fn_island):
            try:
                is_island = bool(str(task.task_id) in set(fn_island()))
            except Exception:
                is_island = False
        map_size = float(max(getattr(self.cfg, "map_size_m", 0.0), 0.0))
        if is_hp or is_island:
            if map_size >= 12000.0:
                if self._legacy_sortie_cap_enabled():
                    sortie_cap = float(max(getattr(self.cfg, "uav_max_sortie_m", long_cap), long_cap))
                    short_cap = float(max(short_cap, min(sortie_cap * 0.45, 3000.0)))
                    long_cap = float(max(long_cap, min(sortie_cap * 0.60, 3600.0)))
                else:
                    short_cap = float(max(short_cap, min(map_size * 0.30, 4200.0)))
                    long_cap = float(max(long_cap, min(map_size * 0.38, 5200.0)))
            elif map_size >= 5000.0:
                if self._legacy_sortie_cap_enabled():
                    sortie_cap = float(max(getattr(self.cfg, "uav_max_sortie_m", long_cap), long_cap))
                    short_cap = float(max(short_cap, min(sortie_cap * 0.35, 1800.0)))
                    long_cap = float(max(long_cap, min(sortie_cap * 0.50, 2400.0)))
                else:
                    short_cap = float(max(short_cap, min(map_size * 0.28, 3000.0)))
                    long_cap = float(max(long_cap, min(map_size * 0.36, 4200.0)))
        if dist_m > float(long_cap):
            return False
        launch_reason = str(reason)
        recovery_buf = float(max(getattr(self.cfg, "uav_recovery_distance_buffer_m", 600.0), 0.0))
        try:
            recovery_buf = float(self._effective_recovery_buffer_for_sortie(str(aid), task, launch_reason=launch_reason))
        except Exception:
            recovery_buf = float(max(getattr(self.cfg, "uav_recovery_distance_buffer_m", 600.0), 0.0))
        mission_chain_m = float(2.0 * max(dist_m, 0.0) + max(recovery_buf, 0.0))
        if self._legacy_sortie_cap_enabled():
            sortie_cap = float(max(getattr(self.cfg, "uav_max_sortie_m", long_cap), long_cap))
            return bool(mission_chain_m <= sortie_cap * 0.92)
        return True

    def _uav_transfer_target_for_task(self, aid: str, task: Optional[DeliveryTask] = None) -> Optional[str]:
        uid = str(aid)
        target_map = getattr(self, "_uav_transfer_target_truck", {})
        task_map = getattr(self, "_uav_transfer_target_task", {})
        if not isinstance(target_map, dict) or not isinstance(task_map, dict):
            return None
        target_tid = str(target_map.get(uid, "")).strip()
        if not target_tid:
            return None
        target_agent = self.state.agents.get(str(target_tid), None)
        if target_agent is None or target_agent.kind != AgentKind.TRUCK or bool(getattr(target_agent, "crashed", False)):
            return None
        hinted_task_id = str(task_map.get(uid, "")).strip()
        goal_task = task
        if goal_task is None:
            gid = self._effective_goals.get(uid, self._recommended_goals.get(uid, None))
            goal_task = self.state.tasks.get(str(gid)) if gid is not None else None
        if goal_task is None or goal_task.kind != TaskKind.EMERGENCY or goal_task.status != TaskStatus.PENDING:
            return None
        if hinted_task_id and hinted_task_id != str(goal_task.task_id):
            return None
        # A task-bound transfer must make geometric progress toward that task.
        # If the published receiver is worse than the current carrier, select
        # the closest physically feasible receiver instead.  This changes only
        # the UAV handoff target; truck routes remain untouched.
        state = self.state.agents.get(uid)
        current_tid = str(getattr(state, "follow_target", "") or "")
        if state is not None and current_tid:
            task_xy = self._node_xy(int(goal_task.demand_node))
            current_xy = self._agent_xy(current_tid)
            planned_xy = self._agent_xy(str(target_tid))
            current_task_dist = float(
                np.hypot(
                    float(current_xy[0]) - float(task_xy[0]),
                    float(current_xy[1]) - float(task_xy[1]),
                )
            )
            planned_task_dist = float(
                np.hypot(
                    float(planned_xy[0]) - float(task_xy[0]),
                    float(planned_xy[1]) - float(task_xy[1]),
                )
            )
            tolerance = float(
                max(getattr(self.cfg, "uav_delivery_radius_m", 100.0), 100.0)
            )
            if (
                _UAV_TRANSFER_RECEIVER_PROGRESS_OVERRIDE_ENABLED
                and planned_task_dist > current_task_dist + tolerance
            ):
                candidates = []
                for candidate_id, candidate in self.state.agents.items():
                    cid = str(candidate_id)
                    if (
                        candidate.kind != AgentKind.TRUCK
                        or cid == current_tid
                        or bool(getattr(candidate, "crashed", False))
                        or not self._truck_has_follow_slot(
                            cid, exclude_aid=uid
                        )
                        or not self._truck_can_accept_uav_payload(cid, uid)
                    ):
                        continue
                    candidate_xy = self._agent_xy(cid)
                    candidate_task_dist = float(
                        np.hypot(
                            float(candidate_xy[0]) - float(task_xy[0]),
                            float(candidate_xy[1]) - float(task_xy[1]),
                        )
                    )
                    if candidate_task_dist >= current_task_dist - tolerance:
                        continue
                    transfer_ok, _ = self._uav_transfer_takeoff_gate_check(
                        uid, cid, task=goal_task
                    )
                    if transfer_ok:
                        candidates.append((candidate_task_dist, cid))
                if candidates:
                    _, target_tid = min(candidates)
                    self._uav_transfer_target_truck[uid] = str(target_tid)
        return str(target_tid)

    def _uav_transfer_takeoff_gate_check(
        self,
        aid: str,
        target_truck_id: str,
        task: Optional[DeliveryTask] = None,
    ) -> Tuple[bool, str]:
        s = self.state.agents.get(str(aid), None)
        ts = self.state.agents.get(str(target_truck_id), None)
        if s is None or s.kind != AgentKind.UAV:
            return False, "not_uav"
        if s.follow_target is None:
            return False, "not_docked"
        if ts is None or ts.kind != AgentKind.TRUCK or bool(getattr(ts, "crashed", False)):
            return False, "target_truck_invalid"
        if str(s.follow_target) == str(target_truck_id):
            return False, "already_on_target_truck"
        if bool(getattr(s, "uav_needs_reload_flag", False)) or (not bool(self._uav_loaded(str(aid)))):
            return False, "not_loaded"

        min_batt = float(np.clip(getattr(self.cfg, "uav_transfer_min_battery_fraction", 0.50), 0.0, 1.0))
        reserve = float(np.clip(getattr(self.cfg, "uav_transfer_reserve_fraction", 0.08), 0.0, 1.0))
        batt = float(max(getattr(s, "battery", 0.0), 0.0))
        if batt + 1e-9 < min_batt:
            return False, "transfer_below_min"

        sxy = s.pos_xy if s.pos_xy is not None else self._node_xy(int(s.node or 0))
        txy = ts.pos_xy if ts.pos_xy is not None else self._node_xy(int(ts.node or 0))
        dist_to_target = float(np.hypot(float(sxy[0]) - float(txy[0]), float(sxy[1]) - float(txy[1])))
        max_target_dist = float(max(getattr(self.cfg, "hrl_uav_task_transfer_max_target_dist_m", 2600.0), 0.0))
        if max_target_dist > 0.0 and dist_to_target > max_target_dist:
            return False, "target_truck_too_far"

        bind_window = float(max(self._uav_bind_window_m(ts), 1.0))
        truck_speed = float(max(getattr(self.cfg, "truck_speed_mps", 0.0), 0.0))
        decision_interval = int(max(getattr(self.cfg, "decision_interval", 5), 1))
        drift_margin = float(truck_speed * self._dt_seconds * decision_interval + bind_window)
        req_transfer = float(
            self._uav_energy_cost_fraction(
                str(aid),
                float(dist_to_target + drift_margin),
                sxy,
                destination=self._uav_extended_destination(
                    sxy, txy, float(dist_to_target + drift_margin)
                ),
                payload_override=float(getattr(s, "payload_kg_current", 0.0)),
            )
            + reserve
        )
        if batt + 1e-9 < req_transfer:
            return False, "transfer_insufficient_margin"
        return True, "truck_transfer"

    def _uav_launch_gate_check(
        self,
        aid: str,
        task: Optional[DeliveryTask] = None,
        *,
        count_reject: bool = False,
    ) -> Tuple[bool, str, bool]:
        def _ret(ok: bool, reason: str, force_recovery: bool) -> Tuple[bool, str, bool]:
            self._record_uav_launch_gate_result(bool(ok), str(reason))
            return bool(ok), str(reason), bool(force_recovery)

        s = self.state.agents[str(aid)]
        if s.kind != AgentKind.UAV:
            if count_reject:
                self._note_uav_task_reject(str(aid), task, "not_uav")
            return _ret(False, "not_uav", False)
        if s.follow_target is None:
            if count_reject:
                self._note_uav_task_reject(str(aid), task, "not_docked")
            return _ret(False, "not_docked", False)

        batt = float(max(getattr(s, "battery", 0.0), 0.0))

        gid = self._effective_goals.get(str(aid), self._recommended_goals.get(str(aid), None))
        task_eff = task if task is not None else (self.state.tasks.get(str(gid)) if gid is not None else None)
        task = task_eff
        if task is None or not self._task_is_uav_delivery(task) or task.status != TaskStatus.PENDING:
            if count_reject:
                self._note_uav_task_reject(str(aid), task_eff, "no_emergency_goal")
            return _ret(False, "no_emergency_goal", True)

        v2_launch = v2_authoritative_launch_check(self, str(aid), task)
        if v2_launch is not None:
            ok_v2, reason_v2, force_recovery_v2 = v2_launch
            if count_reject and not ok_v2:
                self._note_uav_task_reject(str(aid), task, str(reason_v2))
            return _ret(bool(ok_v2), str(reason_v2), bool(force_recovery_v2))

        if not self._uav_loaded_for_task(str(aid), task):
            if count_reject:
                self._note_uav_task_reject(str(aid), task_eff, "not_loaded")
            return _ret(False, "not_loaded", True)

        cur_xy = s.pos_xy if s.pos_xy is not None else self._node_xy(int(s.node or 0))
        tnode = self.topology.nodes[int(task.demand_node)]
        txy = (float(tnode.x), float(tnode.y))
        launch_weather_reason = self._uav_weather_safety_reason(cur_xy)
        if not launch_weather_reason:
            launch_weather_reason = self._uav_weather_safety_reason(txy)
        if launch_weather_reason:
            if count_reject:
                self._note_uav_task_reject(str(aid), task, launch_weather_reason)
            # A docked UAV waits when launch conditions are unsafe.  Forced
            # recovery is reserved for an aircraft that is already airborne.
            return _ret(False, launch_weather_reason, False)
        d_go = float(np.hypot(float(cur_xy[0]) - txy[0], float(cur_xy[1]) - txy[1]))
        back_tid, d_back = self._nearest_truck_from_xy(txy)
        if not np.isfinite(d_back):
            if count_reject:
                self._note_uav_task_reject(str(aid), task, "corridor")
            return _ret(False, "no_truck_for_return", True)

        reserve = float(np.clip(getattr(self.cfg, "uav_emergency_reserve_fraction", 0.20), 0.0, 1.0))
        return_margin = float(np.clip(getattr(self.cfg, "uav_return_margin_fraction", 0.15), 0.0, 1.0))
        rendez_margin = float(np.clip(getattr(self.cfg, "uav_rendezvous_margin_fraction", 0.10), 0.0, 1.0))
        margin_scale = float(np.clip(getattr(self.cfg, "uav_launch_recovery_margin_scale", 1.0), 0.2, 1.5))
        direct_recovery_buf = float(self._effective_recovery_buffer_for_sortie(str(aid), task, launch_reason="direct_safe"))

        truck_speed = float(max(getattr(self.cfg, "truck_speed_mps", 0.0), 0.0))
        decision_interval = int(max(getattr(self.cfg, "decision_interval", 5), 1))
        drift_raw = float(truck_speed * self._dt_seconds * decision_interval)
        drift_scale = float(max(getattr(self.cfg, "uav_recovery_truck_drift_margin_scale", 0.5), 0.0))
        drift_cap = float(max(getattr(self.cfg, "uav_recovery_truck_drift_margin_max_m", 600.0), 0.0))
        truck_drift_margin_m = float(min(drift_raw * drift_scale, drift_cap))

        req_go = float(
            self._uav_energy_cost_fraction(
                str(aid),
                d_go,
                cur_xy,
                destination=txy,
                payload_override=float(getattr(s, "payload_kg_current", 0.0)),
            )
        )

        recovery_xy = txy
        if back_tid is not None and back_tid in self.state.agents:
            back_state = self.state.agents[str(back_tid)]
            recovery_xy = back_state.pos_xy if back_state.pos_xy is not None else self._node_xy(int(back_state.node or 0))
        direct_back_dist = float(d_back + margin_scale * (direct_recovery_buf + truck_drift_margin_m))

        req_back_direct = float(
            self._uav_energy_cost_fraction(
                str(aid),
                direct_back_dist,
                txy,
                destination=self._uav_extended_destination(txy, recovery_xy, direct_back_dist),
                payload_override=self._uav_expected_payload_after_task(str(aid), task),
            )
        )
        direct_req = float(req_go + req_back_direct + reserve + margin_scale * (return_margin + 0.5 * rendez_margin))
        if batt + 1e-9 >= direct_req:
            return _ret(True, "direct_safe", False)

        rendezvous_allowed = bool(getattr(self.cfg, "truck_support_uav_recovery_enabled", True))
        allow_rendezvous_launch = bool(getattr(self.cfg, "uav_allow_rendezvous_launch", False))
        if allow_rendezvous_launch and bool(getattr(self.cfg, "uav_rendezvous_launch_requires_docked_truck_goal", False)):
            truck_id = str(getattr(s, "follow_target", "")) if getattr(s, "follow_target", None) is not None else ""
            truck_goal = self._effective_goals.get(truck_id, self._recommended_goals.get(truck_id, None)) if truck_id else None
            allow_rendezvous_launch = bool(truck_goal is not None and str(truck_goal) == str(getattr(task, "task_id", "")))
        is_island_goal = bool(self._task_is_island(task))
        is_high_pressure = bool(self._is_high_pressure_emergency_task(task) or self._is_high_pressure_island_task(task))

        bind_window = float(max(self._uav_bind_window_m(), 1.0))
        std_buf = float(self._effective_recovery_buffer_for_sortie(str(aid), task, launch_reason="rendezvous_safe"))
        rendez_recovery_dist = float(max(0.75 * std_buf, bind_window))
        rendez_recovery_req = float(
            self._uav_energy_cost_fraction(
                str(aid),
                float(margin_scale * (rendez_recovery_dist + truck_drift_margin_m)),
                txy,
                destination=self._uav_extended_destination(
                    txy,
                    recovery_xy,
                    float(margin_scale * (rendez_recovery_dist + truck_drift_margin_m)),
                ),
                payload_override=self._uav_expected_payload_after_task(str(aid), task),
            )
        )
        rendez_req = float(req_go + rendez_recovery_req + reserve + margin_scale * rendez_margin)

        if rendezvous_allowed and allow_rendezvous_launch and batt + 1e-9 >= rendez_req:
            return _ret(True, "rendezvous_safe", True)
        if rendezvous_allowed and (not allow_rendezvous_launch) and is_island_goal and batt + 1e-9 >= rendez_req:
            return _ret(True, "rendezvous_safe_island", True)

        if (
            rendezvous_allowed
            and (not allow_rendezvous_launch)
            and bool(getattr(self.cfg, "uav_conditional_rendezvous_launch_enabled", True))
            and batt + 1e-9 >= rendez_req
        ):
            pending_emergency = int(self._pending_emergency_task_count())
            min_pending = int(max(getattr(self.cfg, "uav_conditional_rendezvous_min_pending_emergency", 4), 0))
            slack = int(max(int(getattr(task, "deadline_step", self.cfg.max_steps)) - int(self.state.step_index), 0))
            max_slack = int(max(getattr(self.cfg, "uav_conditional_rendezvous_max_deadline_slack_steps", 18), 0))
            nearest_truck_max = float(max(getattr(self.cfg, "uav_conditional_rendezvous_max_nearest_truck_m", 1400.0), 0.0))
            if pending_emergency >= min_pending and slack <= max_slack and float(d_back) <= nearest_truck_max:
                return _ret(True, "rendezvous_safe_relaxed", True)

        if (
            rendezvous_allowed
            and (not allow_rendezvous_launch)
            and bool(getattr(self.cfg, "uav_high_pressure_rendezvous_enabled", True))
            and bool(is_high_pressure)
        ):
            slack = int(max(int(getattr(task, "deadline_step", self.cfg.max_steps)) - int(self.state.step_index), 0))
            hp_max_slack = int(max(getattr(self.cfg, "uav_high_pressure_rendezvous_max_deadline_slack_steps", 28), 0))
            hp_nearest_truck_max = float(max(getattr(self.cfg, "uav_high_pressure_rendezvous_max_nearest_truck_m", 2200.0), 0.0))
            corridor_ok = bool(np.isfinite(d_back) and float(d_back) <= hp_nearest_truck_max)
            relaxed_buf = float(
                self._effective_recovery_buffer_for_sortie(str(aid), task, launch_reason="rendezvous_safe_relaxed_hp")
            )
            relaxed_recovery_dist = float(max(0.75 * relaxed_buf, bind_window))
            relaxed_recovery_req = float(
                self._uav_energy_cost_fraction(
                    str(aid),
                    float(margin_scale * (relaxed_recovery_dist + truck_drift_margin_m)),
                    txy,
                    destination=self._uav_extended_destination(
                        txy,
                        recovery_xy,
                        float(margin_scale * (relaxed_recovery_dist + truck_drift_margin_m)),
                    ),
                    payload_override=self._uav_expected_payload_after_task(str(aid), task),
                )
            )
            relaxed_req = float(req_go + relaxed_recovery_req + reserve + margin_scale * rendez_margin)
            if slack <= hp_max_slack and corridor_ok and batt + 1e-9 >= relaxed_req:
                return _ret(True, "rendezvous_safe_relaxed_hp", True)
            if slack <= hp_max_slack and (not corridor_ok):
                if count_reject:
                    self._note_uav_task_reject(str(aid), task, "corridor")
                return _ret(False, "corridor_blocked", True)

        if rendezvous_allowed and (not allow_rendezvous_launch) and batt + 1e-9 >= rendez_req:
            if count_reject:
                self._note_uav_task_reject(str(aid), task, "rendezvous_launch_disabled")
            return _ret(False, "rendezvous_launch_disabled", True)

        if count_reject:
            self._note_uav_task_reject(str(aid), task, "insufficient_recovery_margin")
        return _ret(False, "insufficient_recovery_margin", True)

    def _uav_sortie_chain_actionable_from_truck_node(self, aid: str, truck_node: int, task: DeliveryTask) -> bool:
        s = self.state.agents.get(str(aid), None)
        if s is None or s.kind != AgentKind.UAV or bool(getattr(s, "crashed", False)):
            return False
        if task is None or task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
            return False
        if bool(getattr(s, "uav_needs_reload_flag", False)):
            return False
        if not bool(self._uav_loaded(str(aid))):
            return False
        tx, ty = self._node_xy(int(truck_node))
        node = self.topology.nodes[int(task.demand_node)]
        dist_m = float(np.hypot(float(node.x) - float(tx), float(node.y) - float(ty)))
        if not np.isfinite(dist_m):
            return False
        short_cap = float(max(getattr(self.cfg, "uav_short_sortie_max_distance_m", 1200.0), 1.0))
        long_cap = float(max(short_cap * 1.60, short_cap))
        is_hp = False
        fn_hp = getattr(self, "_is_high_pressure_emergency_task", None)
        if callable(fn_hp):
            try:
                is_hp = bool(fn_hp(task))
            except Exception:
                is_hp = False
        is_island = bool(self._task_is_island(task))
        map_size = float(max(getattr(self.cfg, "map_size_m", 0.0), 0.0))
        if is_hp or is_island:
            if map_size >= 12000.0:
                if self._legacy_sortie_cap_enabled():
                    sortie_cap = float(max(getattr(self.cfg, "uav_max_sortie_m", long_cap), long_cap))
                    short_cap = float(max(short_cap, min(sortie_cap * 0.45, 3000.0)))
                    long_cap = float(max(long_cap, min(sortie_cap * 0.60, 3600.0)))
                else:
                    short_cap = float(max(short_cap, min(map_size * 0.30, 4200.0)))
                    long_cap = float(max(long_cap, min(map_size * 0.38, 5200.0)))
            elif map_size >= 5000.0:
                if self._legacy_sortie_cap_enabled():
                    sortie_cap = float(max(getattr(self.cfg, "uav_max_sortie_m", long_cap), long_cap))
                    short_cap = float(max(short_cap, min(sortie_cap * 0.35, 1800.0)))
                    long_cap = float(max(long_cap, min(sortie_cap * 0.50, 2400.0)))
                else:
                    short_cap = float(max(short_cap, min(map_size * 0.28, 3000.0)))
                    long_cap = float(max(long_cap, min(map_size * 0.36, 4200.0)))
        if dist_m > float(long_cap):
            return False
        recovery_buf = float(max(getattr(self.cfg, "uav_recovery_distance_buffer_m", 600.0), 0.0))
        mission_chain_m = float(2.0 * max(dist_m, 0.0) + max(recovery_buf, 0.0))
        if self._legacy_sortie_cap_enabled():
            sortie_cap = float(max(getattr(self.cfg, "uav_max_sortie_m", long_cap), long_cap))
        else:
            sortie_cap = float("inf")
        if mission_chain_m > sortie_cap * 0.92:
            return False
        batt = float(getattr(s, "battery", 0.0))
        launch_xy = (float(tx), float(ty))
        task_xy = (float(node.x), float(node.y))
        return_dist = float(max(dist_m, 0.0) + max(recovery_buf, 0.0))
        req = float(
            self._uav_energy_cost_fraction(
                str(aid), dist_m, launch_xy, destination=task_xy,
                payload_override=float(getattr(s, "payload_kg_current", 0.0)),
            )
            + self._uav_energy_cost_fraction(
                str(aid), return_dist, task_xy,
                destination=self._uav_extended_destination(task_xy, launch_xy, return_dist),
                payload_override=self._uav_expected_payload_after_task(str(aid), task),
            )
        )
        return bool(np.isfinite(req) and batt + 1e-9 >= req)

    def _truck_island_forward_support_target(
        self,
        aid: str,
        neighbors: List[int],
        island_task_ids: set,
    ) -> Optional[int]:
        if not neighbors or (not island_task_ids):
            return None
        truck = self.state.agents.get(str(aid), None)
        if truck is None or truck.kind != AgentKind.TRUCK or bool(getattr(truck, "crashed", False)):
            return None
        if truck.node is None:
            return None

        island_nodes: List[int] = []
        for tid in sorted(island_task_ids):
            task = self.state.tasks.get(str(tid), None)
            if task is None or task.kind != TaskKind.EMERGENCY:
                continue
            if task.status not in (TaskStatus.PENDING, TaskStatus.CLAIMED):
                continue
            island_nodes.append(int(task.demand_node))
        if not island_nodes:
            return None

        cur_xy = self._node_xy(int(truck.node))
        cur_best = min(
            float(np.hypot(float(self._node_xy(n)[0]) - float(cur_xy[0]), float(self._node_xy(n)[1]) - float(cur_xy[1])))
            for n in island_nodes
        )

        follower_uavs = []
        for uid, us in self.state.agents.items():
            if us.kind != AgentKind.UAV or bool(getattr(us, "crashed", False)):
                continue
            if str(getattr(us, "follow_target", "")) != str(aid):
                continue
            if bool(getattr(us, "uav_needs_reload_flag", False)):
                continue
            if not bool(self._uav_loaded(str(uid))):
                continue
            follower_uavs.append(str(uid))

        best_nb: Optional[int] = None
        best_key = None
        for nb in neighbors:
            nxy = self._node_xy(int(nb))
            dnb = min(
                float(np.hypot(float(self._node_xy(n)[0]) - float(nxy[0]), float(self._node_xy(n)[1]) - float(nxy[1])))
                for n in island_nodes
            )
            actionable_count = 0
            for tid in sorted(island_task_ids):
                task = self.state.tasks.get(str(tid), None)
                if task is None or task.kind != TaskKind.EMERGENCY or task.status not in (TaskStatus.PENDING, TaskStatus.CLAIMED):
                    continue
                if any(self._uav_sortie_chain_actionable_from_truck_node(str(uid), int(nb), task) for uid in follower_uavs):
                    actionable_count += 1
            dist_gain = float(max(cur_best - dnb, 0.0))
            key = (-int(actionable_count), -float(dist_gain), float(dnb), int(nb))
            if best_key is None or key < best_key:
                best_key = key
                best_nb = int(nb)

        if best_nb is None:
            return None
        best_nb_xy = self._node_xy(int(best_nb))
        best_nb_dist = min(
            float(np.hypot(float(self._node_xy(n)[0]) - float(best_nb_xy[0]), float(self._node_xy(n)[1]) - float(best_nb_xy[1])))
            for n in island_nodes
        )
        # Allow limited non-monotonic detour to escape local geometric traps
        # when road constraints require a temporary lateral move.
        slack_m = float(max(getattr(self.cfg, "truck_island_support_nonmonotonic_slack_m", 450.0), 0.0))
        if best_nb_dist > float(cur_best + slack_m):
            return None
        return int(best_nb)

    def _truck_shared_map_relevant_frontier_target(self, aid: str, neighbors: List[int]) -> Optional[int]:
        if not neighbors:
            return None
        if not self._decision_mode_shared():
            return None
        truck = self.state.agents.get(str(aid), None)
        if truck is None or truck.kind != AgentKind.TRUCK or bool(getattr(truck, "crashed", False)):
            return None
        if truck.node is None:
            return None

        # Shared-map frontier fallback: when cognition changed and current tasking is
        # unavailable, move toward neighbors that improve decision-graph reachability.
        map_changed = bool(
            bool(getattr(self, "_shared_map_update_event_step", False))
            or int(getattr(self, "_shared_map_new_blocked_step", 0)) > 0
            or int(getattr(self, "_unknown_blocked_edge_hit_step", 0)) > 0
        )
        if (not map_changed) and float(self._decision_blocked_ratio()) <= 0.02:
            return None

        pending = [
            t
            for t in self.state.tasks.values()
            if t.status in (TaskStatus.PENDING, TaskStatus.CLAIMED)
        ]
        if not pending:
            return None

        cur_node = int(truck.node)
        dist_norm = float(max(getattr(self.cfg, "pbrs_distance_norm_m", 3000.0), 1e-6))
        best_nb: Optional[int] = None
        best_score = -1e9
        for nb in neighbors:
            nb_i = int(nb)
            score_nb = 0.0
            for t in pending:
                tnode = int(t.demand_node)
                cur_d = float(self._decision_shortest_path_distance(cur_node, tnode))
                nb_d = float(self._decision_shortest_path_distance(nb_i, tnode))
                if not np.isfinite(nb_d):
                    continue
                if not np.isfinite(cur_d):
                    gain = 1.0
                else:
                    gain = float(max(cur_d - nb_d, 0.0) / dist_norm)
                if t.kind == TaskKind.EMERGENCY:
                    gain *= 1.15
                score_nb = max(score_nb, float(gain))
            if score_nb > best_score + 1e-12:
                best_score = float(score_nb)
                best_nb = nb_i

        if best_nb is None or best_score <= 0.0:
            return None
        return int(best_nb)

    def _truck_recovery_support_target(self, aid: str, neighbors: List[int]) -> Optional[int]:
        if not bool(getattr(self.cfg, "truck_support_uav_recovery_enabled", True)):
            return None
        if not neighbors:
            return None
        truck = self.state.agents.get(str(aid), None)
        if truck is None or truck.kind != AgentKind.TRUCK or bool(getattr(truck, "crashed", False)):
            return None
        txy = truck.pos_xy if truck.pos_xy is not None else self._node_xy(int(truck.node or 0))

        candidates = []
        req_map = getattr(self, "_uav_recovery_requested_truck", {}) if hasattr(self, "_uav_recovery_requested_truck") else {}
        request_bonus = float(max(getattr(self.cfg, "truck_recovery_request_match_bonus", 0.35), 0.0))
        force_thr = float(np.clip(getattr(self.cfg, "uav_low_battery_force_recover_threshold", 0.25), 0.0, 1.0))
        for uid, us in self.state.agents.items():
            if us.kind != AgentKind.UAV or bool(getattr(us, "crashed", False)):
                continue
            if us.follow_target is not None:
                continue

            requested_tid = str(req_map.get(str(uid))) if isinstance(req_map, dict) and str(uid) in req_map else None
            request_match = bool(requested_tid is not None and requested_tid == str(aid))
            hard_recovery = bool(self._uav_hard_recovery_required(str(uid)))
            needs = bool(
                hard_recovery
                or bool(getattr(us, "uav_needs_reload_flag", False))
                or (not bool(self._uav_loaded(str(uid))))
                or request_match
            )
            if not needs:
                continue

            uxy = us.pos_xy if us.pos_xy is not None else self._node_xy(int(us.node or 0))
            vx, vy = us.vel_xy if us.vel_xy is not None else (0.0, 0.0)
            pred_xy = (
                float(np.clip(float(uxy[0]) + float(vx) * self._dt_seconds, 0.0, float(self.cfg.map_size_m))),
                float(np.clip(float(uxy[1]) + float(vy) * self._dt_seconds, 0.0, float(self.cfg.map_size_m))),
            )
            d_now = float(np.hypot(float(pred_xy[0]) - float(txy[0]), float(pred_xy[1]) - float(txy[1])))

            batt = float(getattr(us, "battery", 0.0))
            batt_urg = float(max(force_thr - batt, 0.0) / max(force_thr, 1e-6))
            needs_reload = bool(getattr(us, "uav_needs_reload_flag", False))
            unloaded = not bool(self._uav_loaded(str(uid)))
            forced_latch = bool(
                self._uav_forced_rth_latch.get(str(uid), False)
                and hard_recovery
            )
            low_thr = float(np.clip(getattr(self.cfg, "uav_low_battery_goal_lock_threshold", 0.35), 0.0, 1.0))
            battery_critical = bool(batt <= force_thr)
            reload_low = bool((needs_reload or unloaded) and batt <= low_thr)
            planned_recovery = bool(request_match and bool(getattr(self.cfg, "uav_rendezvous_planned_recovery_request_enabled", True)))
            needs = bool(battery_critical or forced_latch or reload_low or planned_recovery)
            if not needs:
                continue

            urgency = float(batt_urg)
            if planned_recovery:
                urgency = float(max(urgency, np.clip(getattr(self.cfg, "uav_rendezvous_planned_recovery_urgency", 0.55), 0.0, 1.0)))
            if reload_low:
                urgency = float(max(urgency, 0.75))
            if forced_latch:
                urgency = float(max(urgency, 0.90))
            if batt <= max(0.10, 0.6 * force_thr):
                urgency = float(max(urgency, 1.00))
            if request_match and not planned_recovery:
                urgency = float(max(urgency, 0.95))

            candidates.append((str(uid), pred_xy, d_now, urgency, request_match))

        if not candidates:
            return None
        if (
            bool(getattr(self.cfg, "truck_recovery_require_request_when_normal_pending", False))
            and sum(1 for t in self.state.tasks.values() if t.status == TaskStatus.PENDING and t.kind == TaskKind.NORMAL) > 0
            and self._truck_has_reachable_serviceable_normal(str(aid))
        ):
            min_urg = float(np.clip(getattr(self.cfg, "truck_recovery_request_min_urgency_when_normal_pending", 0.0), 0.0, 1.0))
            candidates = [c for c in candidates if bool(c[4])]
            if min_urg > 0.0:
                candidates = [c for c in candidates if float(c[3]) >= min_urg]
            if not candidates:
                return None

        hard_urgent = [c for c in candidates if float(c[3]) >= 0.75]
        if hard_urgent:
            hard_urgent.sort(key=lambda x: (-float(x[3]), float(x[2]), str(x[0])))
            _, hard_xy, hard_d_now, _, _ = hard_urgent[0]
            best_nb_hard = min(
                neighbors,
                key=lambda nb: float(
                    np.hypot(
                        float(self._node_xy(int(nb))[0]) - float(hard_xy[0]),
                        float(self._node_xy(int(nb))[1]) - float(hard_xy[1]),
                    )
                ),
            )
            best_nb_hard_xy = self._node_xy(int(best_nb_hard))
            hard_d_nb = float(
                np.hypot(
                    float(best_nb_hard_xy[0]) - float(hard_xy[0]),
                    float(best_nb_hard_xy[1]) - float(hard_xy[1]),
                )
            )
            if hard_d_nb + 1e-6 < float(hard_d_now):
                return int(best_nb_hard)
            return None

        best_nb = None
        best_score = -1e9
        dist_norm = float(max(getattr(self.cfg, "pbrs_distance_norm_m", 3000.0), 1e-6))
        detour_w = float(max(getattr(self.cfg, "truck_recovery_max_detour_cost_weight", 1.0), 0.0))
        pri_w = float(max(getattr(self.cfg, "truck_recovery_priority_weight", 1.0), 0.0))

        cur_task = self._pbrs_target_task(str(aid))
        best_goal_dist = None
        if cur_task is not None:
            goal_node = int(cur_task.demand_node)
            goal_dists = [
                float(self._decision_shortest_path_distance(int(nb), goal_node))
                for nb in neighbors
            ]
            finite_goal = [d for d in goal_dists if np.isfinite(d)]
            if finite_goal:
                best_goal_dist = float(min(finite_goal))

        for nb in neighbors:
            nxy = self._node_xy(int(nb))
            support_score = 0.0
            for _, pred_xy, d_now, urgency, request_match in candidates:
                d_nb = float(np.hypot(float(pred_xy[0]) - float(nxy[0]), float(pred_xy[1]) - float(nxy[1])))
                reduction = float(d_now - d_nb)
                base = float(pri_w * (float(reduction / max(d_now, 1e-6)) + float(urgency)))
                if request_match:
                    base += float(request_bonus)
                support_score = max(support_score, base)

            detour_pen = 0.0
            if cur_task is not None and best_goal_dist is not None:
                goal_d = float(self._decision_shortest_path_distance(int(nb), int(cur_task.demand_node)))
                if np.isfinite(goal_d):
                    detour_pen = float(max(goal_d - best_goal_dist, 0.0) / dist_norm)

            score = float(support_score - detour_w * detour_pen)
            if score > best_score + 1e-12:
                best_score = score
                best_nb = int(nb)

        if best_score <= 0.0:
            return None
        return best_nb

    def _truck_can_service_task(self, aid: str, task: DeliveryTask) -> bool:
        s = self.state.agents[str(aid)]
        if s.kind != AgentKind.TRUCK:
            return False
        if not self._vehicle_compatibility(task, AgentKind.TRUCK):
            return False
        self._sync_truck_inventory_fields(s)
        if s.node is None:
            return False

        # Truck serviceability must be road-feasible under the current decision graph.
        # This prevents assigning unreachable NORMAL tasks that cause truck ping-pong.
        route_dist = float(self._decision_shortest_path_distance(int(s.node), int(task.demand_node)))
        route_reachable = bool(np.isfinite(route_dist))

        req_units = int(self._task_supply_units_required(task))
        req_tc_kg = float(max(float(req_units) * self._timecritical_supply_unit_kg(), 1e-6))
        if self._task_supply_type(task) == "emergency":
            serviceable = False
            # Must have time-critical stock first.
            if float(getattr(s, "timecritical_inventory_kg_current", 0.0)) >= req_tc_kg - 1e-9:
                if bool(getattr(self.cfg, "truck_can_serve_emergency_tasks", False)):
                    serviceable = bool(route_reachable)
                elif bool(getattr(self.cfg, "truck_conditional_emergency_service_enabled", True)):
                    # Controlled fallback: allow only near+urgent emergency service,
                    # and only when route is actually reachable.
                    if route_reachable:
                        max_dist = float(max(getattr(self.cfg, "truck_emergency_service_max_distance_m", 900.0), 0.0))
                        max_slack = int(max(getattr(self.cfg, "truck_emergency_service_max_deadline_slack_steps", 18), 0))
                        # High-pressure L/island fallback widens emergency service window.
                        if bool(self._is_high_pressure_emergency_task(task) or self._is_high_pressure_island_task(task)):
                            max_dist = float(max(max_dist, float(max(getattr(self.cfg, "truck_high_pressure_emergency_service_max_distance_m", 1400.0), 0.0))))
                            max_slack = int(max(max_slack, int(max(getattr(self.cfg, "truck_high_pressure_emergency_service_max_deadline_slack_steps", 28), 0))))
                        slack = int(max(int(getattr(task, "deadline_step", self.cfg.max_steps)) - int(self.state.step_index), 0))
                        # Conditional eligibility follows the preregistered
                        # near-or-urgent rule: a reachable truck may serve TC
                        # when either road distance or remaining slack crosses
                        # its threshold.
                        serviceable = bool(route_dist <= max_dist or slack <= max_slack)
            self._note_truck_emergency_serviceability(str(aid), task, bool(serviceable))
            return bool(serviceable)

        return bool(float(getattr(s, "bulk_inventory_kg_current", 0.0)) > 1e-9 and route_reachable)

    def _vehicle_compatibility(self, task: DeliveryTask, agent_kind: AgentKind) -> bool:
        # Semantics-first compatibility:
        # - routine_bulk: primarily truck service
        # - time_critical_lightweight: UAV-preferred, truck fallback may be allowed by config.
        if agent_kind == AgentKind.UAV:
            return bool(
                self._task_is_time_critical_lightweight(task)
                or self._task_is_bulk_relay(task)
            )
        if agent_kind == AgentKind.TRUCK:
            if self._task_is_routine_bulk(task):
                return True
            return bool(getattr(self.cfg, "truck_can_serve_emergency_tasks", False) or getattr(self.cfg, "truck_conditional_emergency_service_enabled", True))
        return False

    def _uav_can_service_task(self, aid: str, task: DeliveryTask) -> bool:
        s = self.state.agents[str(aid)]
        if s.kind != AgentKind.UAV:
            return False
        if not self._vehicle_compatibility(task, AgentKind.UAV):
            return False
        if bool(getattr(self.cfg, "uav_must_replenish_after_each_service", True)) and bool(
            getattr(s, "uav_needs_reload_flag", False)
        ):
            return False
        return bool(self._uav_loaded_for_task(aid, task))

    def is_task_serviceable_by_agent(self, aid: str, task: DeliveryTask) -> bool:
        a = self.state.agents.get(str(aid), None)
        cooperative_relay_join = bool(
            a is not None
            and a.kind == AgentKind.UAV
            and self._task_is_bulk_relay(task)
            and task.status == TaskStatus.CLAIMED
            and str(aid) in tuple(str(uid) for uid in getattr(task, "route_contract_uav_ids", ()))
            and str(aid) not in tuple(str(uid) for uid in getattr(task, "relay_service_agents", ()))
        )
        if a is None or (task.status != TaskStatus.PENDING and not cooperative_relay_join):
            return False
        if a.kind == AgentKind.UAV:
            return bool(self._uav_can_service_task(str(aid), task))
        if a.kind == AgentKind.TRUCK:
            return bool(self._truck_can_service_task(str(aid), task))
        return False

    def _truck_requires_depot(self, aid: str) -> bool:
        s = self.state.agents.get(str(aid), None)
        if s is None or s.kind != AgentKind.TRUCK:
            return False
        self._sync_truck_inventory_fields(s)
        return bool(getattr(s, "truck_needs_replenish_flag", False))

    def _truck_has_reachable_serviceable_normal(self, aid: str) -> bool:
        s = self.state.agents.get(str(aid), None)
        if s is None or s.kind != AgentKind.TRUCK or bool(getattr(s, "crashed", False)):
            return False
        if s.node is None:
            return False
        for t in self.state.tasks.values():
            if t.status != TaskStatus.PENDING or t.kind != TaskKind.NORMAL:
                continue
            if not bool(self.is_task_serviceable_by_agent(str(aid), t)):
                continue
            d = float(self._decision_shortest_path_distance(int(s.node), int(t.demand_node)))
            if np.isfinite(d):
                return True
        return False

    def _truck_task_priority_score(self, task: DeliveryTask, route_dist_m: float) -> float:
        slack = float(self._task_deadline_slack_steps(task))
        horizon = float(max(getattr(self.cfg, "max_steps", 1), 1))
        urg = float(np.clip(1.0 - slack / horizon, 0.0, 1.0))
        lifeline_init = float(max(getattr(task, "lifeline_init", 100.0), 1e-6))
        lifeline_cur = float(max(getattr(task, "lifeline_current", lifeline_init), 0.0))
        lifeline_urg = float(np.clip(1.0 - lifeline_cur / lifeline_init, 0.0, 1.0))
        cls_bias = 0.10 if task.kind == TaskKind.NORMAL else 0.14
        dist_norm = float(max(getattr(self.cfg, "pbrs_distance_norm_m", 3000.0), 1e-6))
        dist_pen = float(route_dist_m / dist_norm)
        return float(1.25 * urg + 0.55 * lifeline_urg + cls_bias - 0.95 * dist_pen)

    def _truck_best_reachable_service_move(self, aid: str, avoid_node: Optional[int] = None) -> Tuple[Optional[int], bool]:
        s = self.state.agents.get(str(aid), None)
        if s is None or s.kind != AgentKind.TRUCK or bool(getattr(s, "crashed", False)):
            return None, False
        if s.node is None:
            return None, False
        cur = int(s.node)
        neighbors = list(self._decision_neighbors(cur))
        if not neighbors:
            return None, False

        best_task: Optional[DeliveryTask] = None
        best_task_score = -1e18
        has_reachable = False
        for t in self.state.tasks.values():
            if t.status != TaskStatus.PENDING:
                continue
            if not bool(self.is_task_serviceable_by_agent(str(aid), t)):
                continue
            dcur = float(self._decision_shortest_path_distance(cur, int(t.demand_node)))
            if not np.isfinite(dcur):
                continue
            has_reachable = True
            sc = float(self._truck_task_priority_score(t, dcur))
            if sc > best_task_score + 1e-12:
                best_task_score = float(sc)
                best_task = t

        if (not has_reachable) or best_task is None:
            return None, False

        target_node = int(best_task.demand_node)
        cand_neighbors = [int(nb) for nb in neighbors if avoid_node is None or int(nb) != int(avoid_node)]
        if not cand_neighbors:
            cand_neighbors = [int(nb) for nb in neighbors]

        best_nb: Optional[int] = None
        best_nb_dist = float("inf")
        for nb in cand_neighbors:
            dnb = float(self._decision_shortest_path_distance(int(nb), target_node))
            if not np.isfinite(dnb):
                continue
            if dnb + 1e-9 < best_nb_dist:
                best_nb_dist = float(dnb)
                best_nb = int(nb)

        if best_nb is not None:
            return int(best_nb), True

        for t in self.state.tasks.values():
            if t.status != TaskStatus.PENDING:
                continue
            if not bool(self.is_task_serviceable_by_agent(str(aid), t)):
                continue
            for nb in cand_neighbors:
                dnb = float(self._decision_shortest_path_distance(int(nb), int(t.demand_node)))
                if np.isfinite(dnb):
                    return int(nb), True

        return None, bool(has_reachable)
    def _flag_supply_block(self, task: DeliveryTask) -> None:
        if self._task_supply_type(task) == "emergency":
            self.emergency_tasks_blocked_by_supply_count += 1
        else:
            self.normal_tasks_blocked_by_supply_count += 1

    def _consume_supply_for_service(self, aid: str, task: DeliveryTask) -> float:
        s = self.state.agents.get(str(aid), None)
        if s is None:
            return 0.0
        req_units = int(self._task_supply_units_required(task))
        rem_kg = float(max(float(getattr(task, "remaining_demand_kg", self._task_demand_kg(task))), 0.0))
        if rem_kg <= 1e-9:
            return 0.0
        if s.kind == AgentKind.TRUCK:
            if not self._truck_can_service_task(str(aid), task):
                self._flag_supply_block(task)
                s.truck_needs_replenish_flag = True
                return 0.0

            if self._task_supply_type(task) == "emergency":
                tc_unit = float(self._timecritical_supply_unit_kg())
                tc_kg = float(max(getattr(s, "timecritical_inventory_kg_current", 0.0), 0.0))
                # Time-critical single-service package: one 150 kg aviation unit.
                transfer_kg = float(min(rem_kg, float(req_units) * tc_unit, tc_kg))
                s.timecritical_inventory_kg_current = float(max(tc_kg - transfer_kg, 0.0))
            else:
                bulk_kg = float(max(getattr(s, "bulk_inventory_kg_current", 0.0), 0.0))
                if self._task_is_routine_bulk(task) and bool(getattr(self.cfg, "routine_bulk_partial_fulfillment_enabled", True)):
                    chunk = float(max(float(getattr(self.cfg, "routine_bulk_partial_chunk_kg", 100.0)), 1e-6))
                    transfer_kg = float(min(rem_kg, chunk, bulk_kg))
                else:
                    transfer_kg = float(min(rem_kg, bulk_kg))
                s.bulk_inventory_kg_current = float(max(bulk_kg - transfer_kg, 0.0))

            self._sync_truck_inventory_fields(s)
            task.supply_units_consumed = True
            return float(max(transfer_kg, 0.0))

        if s.kind == AgentKind.UAV:
            if not self._uav_can_service_task(str(aid), task):
                self._flag_supply_block(task)
                return 0.0
            payload_kg = float(max(getattr(s, "payload_kg_current", 0.0), 0.0))
            transfer_kg = float(min(rem_kg, payload_kg))
            residual_payload_kg = float(max(payload_kg - transfer_kg, 0.0))
            # Unused material remains physically aboard until the UAV docks;
            # the docked reload stage returns it to the truck before reloading.
            s.carried_emergency_units = int(1 if residual_payload_kg > 1e-9 else 0)
            task.supply_units_consumed = True
            s.payload_kg_current = float(residual_payload_kg)
            s.uav_needs_reload_flag = True
            self._sync_uav_payload_fields(s)
            return float(max(transfer_kg, 0.0))
        return 0.0

    def _solve_min_cost_matching(self, costs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if linear_sum_assignment is not None:
            r, c = linear_sum_assignment(costs)
            return np.asarray(r, dtype=np.int64), np.asarray(c, dtype=np.int64)

        r_sel: List[int] = []
        c_sel: List[int] = []
        used_r = set()
        used_c = set()
        flat = []
        for i in range(costs.shape[0]):
            for j in range(costs.shape[1]):
                flat.append((float(costs[i, j]), i, j))
        flat.sort(key=lambda x: (x[0], x[1], x[2]))
        for _, i, j in flat:
            if i in used_r or j in used_c:
                continue
            used_r.add(i)
            used_c.add(j)
            r_sel.append(int(i))
            c_sel.append(int(j))
        return np.asarray(r_sel, dtype=np.int64), np.asarray(c_sel, dtype=np.int64)

    def _assign_distinct_docked_uav_emergency_goals(self) -> None:
        self.uav_docked_retarget_count_step = 0
        self.uav_urgent_watchdog_assign_count_step = 0
        if not bool(getattr(self.cfg, "uav_docked_retarget_enabled", True)):
            return

        step_now = int(self.state.step_index)
        retarget_interval = int(max(getattr(self.cfg, "uav_docked_retarget_interval_steps", 6), 1))
        initial_enabled = bool(getattr(self.cfg, "uav_initial_distinct_emergency_assign", True))
        initial_window = int(max(getattr(self.cfg, "uav_initial_distinct_window_steps", 6), 0))

        urgent_watchdog_enabled = bool(getattr(self.cfg, "uav_urgent_watchdog_enabled", True))
        urgent_slack_steps = int(max(getattr(self.cfg, "uav_urgent_watchdog_slack_steps", 16), 0))
        urgent_cooldown_steps = int(max(getattr(self.cfg, "uav_urgent_watchdog_retarget_cooldown_steps", 5), 1))
        urgency_bonus_m = float(max(getattr(self.cfg, "uav_urgent_watchdog_distance_bonus_m", 450.0), 0.0))
        urgent_max_assign = int(max(getattr(self.cfg, "uav_urgent_watchdog_max_assign_per_step", 1), 0))
        stale_hold_steps = int(max(2 * retarget_interval, urgent_cooldown_steps + 2, 10))

        for aid0, st0 in self.state.agents.items():
            if st0.kind != AgentKind.UAV:
                continue
            gid0 = self._effective_goals.get(str(aid0), None)
            t0 = self.state.tasks.get(str(gid0), None) if gid0 is not None else None
            active_docked_goal = bool(
                st0.follow_target is not None
                and t0 is not None
                and t0.kind == TaskKind.EMERGENCY
                and t0.status == TaskStatus.PENDING
            )
            if active_docked_goal:
                prev_tid = self._uav_docked_goal_hold_task.get(str(aid0), None)
                if prev_tid == str(gid0):
                    self._uav_docked_goal_hold_steps[str(aid0)] = int(
                        self._uav_docked_goal_hold_steps.get(str(aid0), 0) + 1
                    )
                else:
                    self._uav_docked_goal_hold_task[str(aid0)] = str(gid0)
                    self._uav_docked_goal_hold_steps[str(aid0)] = 1
            else:
                self._uav_docked_goal_hold_task[str(aid0)] = None
                self._uav_docked_goal_hold_steps[str(aid0)] = 0

        def _docked_goal_actionable_now(uid: str, task) -> bool:
            st = self.state.agents.get(str(uid), None)
            if st is None or st.kind != AgentKind.UAV or bool(getattr(st, "crashed", False)):
                return False
            if st.follow_target is None:
                return bool(self.is_task_serviceable_by_agent(str(uid), task))
            if bool(getattr(st, "uav_needs_reload_flag", False)):
                return False
            if not bool(self._uav_loaded(str(uid))):
                return False
            hold_task = self._uav_docked_goal_hold_task.get(str(uid), None)
            hold_steps = int(self._uav_docked_goal_hold_steps.get(str(uid), 0))
            if hold_task == str(task.task_id) and hold_steps >= stale_hold_steps:
                return False
            prev_goal_eval = self._effective_goals.get(str(uid), None)
            try:
                self._effective_goals[str(uid)] = str(task.task_id)
                ok, _, _ = self._uav_launch_gate_check(str(uid), task=task, count_reject=False)
                return bool(ok)
            except Exception:
                return False
            finally:
                self._effective_goals[str(uid)] = prev_goal_eval

        docked_goal_counts: Dict[str, int] = {}
        for aid0, st0 in self.state.agents.items():
            if st0.kind != AgentKind.UAV or bool(getattr(st0, "crashed", False)):
                continue
            if st0.follow_target is None:
                continue
            gid0 = self._effective_goals.get(str(aid0), None)
            t0 = self.state.tasks.get(str(gid0), None) if gid0 is not None else None
            if t0 is None or t0.kind != TaskKind.EMERGENCY or t0.status != TaskStatus.PENDING:
                continue
            if not bool(self._uav_docked_task_actionable_now(str(aid0), t0)):
                continue
            key0 = str(t0.task_id)
            docked_goal_counts[key0] = int(docked_goal_counts.get(key0, 0)) + 1

        occupied_pending_watchdog: set = set()
        for aid0, gid0 in self._effective_goals.items():
            if gid0 is None:
                continue
            t0 = self.state.tasks.get(str(gid0), None)
            if t0 is None:
                continue
            if t0.status == TaskStatus.PENDING and t0.kind == TaskKind.EMERGENCY:
                st0 = self.state.agents.get(str(aid0), None)
                if st0 is not None and st0.kind == AgentKind.UAV and st0.follow_target is not None:
                    if not bool(self._uav_docked_task_actionable_now(str(aid0), t0)):
                        continue
                occupied_pending_watchdog.add(str(t0.task_id))

        urgent_uncovered_ids: List[str] = []
        if urgent_watchdog_enabled and urgent_slack_steps > 0:
            urgent_rows: List[Tuple[int, int, str]] = []
            for tid, task in self.state.tasks.items():
                if task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
                    continue
                tid_s = str(tid)
                if tid_s in occupied_pending_watchdog:
                    continue
                slack = int(max(int(task.deadline_step) - int(step_now), 0))
                if slack <= urgent_slack_steps:
                    urgent_rows.append((int(slack), int(task.deadline_step), tid_s))
            urgent_rows.sort(key=lambda x: (x[0], x[1], x[2]))
            urgent_uncovered_ids = [str(x[2]) for x in urgent_rows]
        urgent_task_set = set(str(x) for x in urgent_uncovered_ids)

        candidate_uavs: List[str] = []
        candidate_due_urgent: Dict[str, bool] = {}
        for aid, st in sorted(self.state.agents.items(), key=lambda kv: str(kv[0])):
            if st.kind != AgentKind.UAV or bool(getattr(st, "crashed", False)):
                continue
            if st.follow_target is None:
                continue
            if bool(getattr(st, "uav_needs_reload_flag", False)):
                continue
            if not bool(self._uav_loaded(str(aid))):
                continue

            last_step = int(self._uav_last_docked_retarget_step.get(str(aid), -10**9))
            due_initial = bool(initial_enabled and step_now <= initial_window)

            current_gid = self._effective_goals.get(str(aid), None)
            current_task = self.state.tasks.get(str(current_gid), None) if current_gid is not None else None
            current_goal_valid = bool(
                current_task is not None
                and current_task.kind == TaskKind.EMERGENCY
                and current_task.status == TaskStatus.PENDING
                and (
                    bool(self._uav_docked_task_actionable_now(str(aid), current_task))
                    if st.follow_target is not None
                    else bool(self.is_task_serviceable_by_agent(str(aid), current_task))
                )
            )
            current_dist = float("inf")
            current_slack = int(10**9)
            if current_goal_valid:
                current_dist = float(self._agent_distance_to_task(str(aid), current_task))
                if not np.isfinite(current_dist):
                    current_goal_valid = False
                else:
                    current_slack = int(max(int(current_task.deadline_step) - int(step_now), 0))

            due_periodic = bool(
                (step_now - last_step) >= retarget_interval
                and (not current_goal_valid)
            )
            due_duplicate_fix = bool(
                current_goal_valid
                and current_gid is not None
                and int(docked_goal_counts.get(str(current_gid), 0)) > 1
            )

            due_urgent = False
            if (
                urgent_watchdog_enabled
                and bool(urgent_uncovered_ids)
                and ((step_now - last_step) >= urgent_cooldown_steps)
            ):
                if not current_goal_valid:
                    due_urgent = True
                elif current_slack > int(urgent_slack_steps + 2):
                    best_urgent_dist = float("inf")
                    for tid in urgent_uncovered_ids:
                        t = self.state.tasks.get(str(tid), None)
                        if t is None:
                            continue
                        if not bool(self.is_task_serviceable_by_agent(str(aid), t)):
                            continue
                        d_u = float(self._agent_distance_to_task(str(aid), t))
                        if np.isfinite(d_u) and d_u < best_urgent_dist:
                            best_urgent_dist = float(d_u)
                    if np.isfinite(best_urgent_dist) and (best_urgent_dist + 30.0 < current_dist):
                        due_urgent = True

            if due_initial or due_duplicate_fix or due_periodic or due_urgent:
                candidate_uavs.append(str(aid))
                candidate_due_urgent[str(aid)] = bool(due_urgent)

        if not candidate_uavs:
            return

        occupied_tasks: set = set()
        for aid, gid in self._effective_goals.items():
            if gid is None or str(aid) in set(candidate_uavs):
                continue
            t = self.state.tasks.get(str(gid), None)
            if t is not None and t.status == TaskStatus.PENDING and t.kind == TaskKind.EMERGENCY:
                occupied_tasks.add(str(t.task_id))

        pending_emergency_ids: List[str] = []
        for tid, task in sorted(self.state.tasks.items(), key=lambda kv: str(kv[0])):
            if task.kind != TaskKind.EMERGENCY or task.status != TaskStatus.PENDING:
                continue
            if str(tid) in occupied_tasks:
                continue
            pending_emergency_ids.append(str(tid))

        if not pending_emergency_ids:
            return

        big = 1e9
        cost = np.full((len(candidate_uavs), len(pending_emergency_ids)), fill_value=big, dtype=np.float64)
        for i, aid in enumerate(candidate_uavs):
            due_urgent = bool(candidate_due_urgent.get(str(aid), False))
            for j, tid in enumerate(pending_emergency_ids):
                task = self.state.tasks.get(str(tid), None)
                if task is None:
                    continue
                if due_urgent and str(tid) not in urgent_task_set:
                    continue
                if not bool(self.is_task_serviceable_by_agent(str(aid), task)):
                    continue

                if due_urgent:
                    prev_goal_eval = self._effective_goals.get(str(aid), None)
                    launch_ok = False
                    try:
                        self._effective_goals[str(aid)] = str(tid)
                        launch_ok, _, _ = self._uav_launch_gate_check(str(aid))
                    except Exception:
                        launch_ok = False
                    finally:
                        self._effective_goals[str(aid)] = prev_goal_eval
                    if not bool(launch_ok):
                        continue

                d = float(self._agent_distance_to_task(str(aid), task))
                if not np.isfinite(d):
                    continue

                d_eff = float(d)
                if due_urgent and urgent_slack_steps > 0:
                    slack = int(max(int(task.deadline_step) - int(step_now), 0))
                    urgency_scale = float(np.clip((float(urgent_slack_steps) - float(slack)) / float(urgent_slack_steps), 0.0, 1.0))
                    d_eff = float(d_eff - urgency_bonus_m * urgency_scale)
                cost[i, j] = float(max(d_eff, 0.0))

        r_sel, c_sel = self._solve_min_cost_matching(cost)
        assigned = 0
        urgent_assigned = 0
        used_tid: set = set()
        for r, c in zip(r_sel.tolist(), c_sel.tolist()):
            if r < 0 or c < 0 or r >= len(candidate_uavs) or c >= len(pending_emergency_ids):
                continue
            if float(cost[int(r), int(c)]) >= big:
                continue
            aid = str(candidate_uavs[int(r)])
            tid = str(pending_emergency_ids[int(c)])
            if tid in used_tid:
                continue

            is_urgent_assign = bool(candidate_due_urgent.get(aid, False) and (tid in urgent_task_set))
            if is_urgent_assign and urgent_assigned >= urgent_max_assign:
                continue

            self._effective_goals[aid] = str(tid)
            self._uav_last_docked_retarget_step[aid] = int(step_now)
            if is_urgent_assign:
                urgent_assigned += 1
            used_tid.add(str(tid))
            assigned += 1

        self.uav_docked_retarget_count_step = int(assigned)
        self.uav_docked_retarget_count_total += int(assigned)
        self.uav_urgent_watchdog_assign_count_step = int(urgent_assigned)
        self.uav_urgent_watchdog_assign_count_total += int(urgent_assigned)

    def _enforce_unique_pending_task_goals(self) -> None:
        # Safety net: keep one agent per pending task goal to avoid duplicate dispatch.
        task_claims: Dict[str, List[str]] = {}
        for aid, gid in self._effective_goals.items():
            if gid is None:
                continue
            task = self.state.tasks.get(str(gid), None)
            if task is None or task.status != TaskStatus.PENDING:
                continue
            task_claims.setdefault(str(task.task_id), []).append(str(aid))

        for tid, claimants in task_claims.items():
            if len(claimants) <= 1:
                continue
            task = self.state.tasks.get(str(tid), None)
            if task is None:
                continue
            # A road-isolated normal task is deliberately shared by the two
            # UAVs named in the same truck contract; all other duplicates are
            # still removed by the ordinary exclusivity guard below.
            cooperative = bool(
                self._task_is_bulk_relay(task)
                and set(str(aid) for aid in claimants).issubset(
                    set(str(uid) for uid in getattr(task, "route_contract_uav_ids", ()))
                )
            )
            if cooperative:
                continue

            best_aid: Optional[str] = None
            best_key: Tuple[float, float, str] = (float('inf'), float('inf'), '')
            contract_owner = str(
                getattr(task, "route_contract_owner", "") or ""
            )
            # The layer-1 owner wins duplicate advertisement while it remains
            # a live claimant. Distance-based arbitration is only a fallback
            # when that versioned owner is absent or invalid.
            if (
                bool(
                    getattr(
                        self.cfg,
                        "hrl_route_plan_atomic_contract_enabled",
                        True,
                    )
                )
                and contract_owner in claimants
                and self.state.agents.get(contract_owner, None) is not None
                and not bool(
                    getattr(self.state.agents[contract_owner], "crashed", False)
                )
            ):
                best_aid = contract_owner
            for aid in claimants:
                if best_aid is not None:
                    break
                st = self.state.agents.get(str(aid), None)
                if st is None or bool(getattr(st, 'crashed', False)):
                    continue
                try:
                    serviceable = bool(self.is_task_serviceable_by_agent(str(aid), task))
                except Exception:
                    serviceable = False
                if not serviceable:
                    role_rank = 3.0
                    dist = float('inf')
                else:
                    if self._task_is_uav_delivery(task):
                        role_rank = 0.0 if st.kind == AgentKind.UAV else 1.0
                    elif task.kind == TaskKind.NORMAL:
                        role_rank = 0.0 if st.kind == AgentKind.TRUCK else 1.0
                    else:
                        role_rank = 1.0
                    dist = float(self._agent_distance_to_task(str(aid), task))
                    if not np.isfinite(dist):
                        dist = float('inf')
                key = (float(role_rank), float(dist), str(aid))
                if key < best_key:
                    best_key = key
                    best_aid = str(aid)

            for aid in claimants:
                if best_aid is None or str(aid) != str(best_aid):
                    self._effective_goals[str(aid)] = None

    def _assign_conditional_truck_emergency_goals(self) -> None:
        """Apply the common near-or-urgent truck fallback used in sensitivity runs.

        The rule is deliberately outside every planner: it changes service
        eligibility, not the optimization logic.  It never interrupts an
        airborne UAV or an in-service task, and it assigns at most one pending
        time-critical task to each road-feasible truck.
        """
        if not bool(getattr(self.cfg, "truck_conditional_emergency_service_enabled", False)):
            return
        if bool(getattr(self.cfg, "truck_can_serve_emergency_tasks", False)):
            return

        protected: set[str] = set()
        for aid, agent in self.state.agents.items():
            if agent.kind != AgentKind.UAV or bool(getattr(agent, "crashed", False)):
                continue
            if getattr(agent, "follow_target", None) is None:
                gid = self._effective_goals.get(str(aid), None)
                if gid is not None:
                    protected.add(str(gid))
                contract_tid = self._uav_sortie_contract_task.get(str(aid), None)
                if contract_tid is not None:
                    protected.add(str(contract_tid))

        candidates: List[DeliveryTask] = []
        for task in self.state.tasks.values():
            if task.status != TaskStatus.PENDING or task.kind != TaskKind.EMERGENCY:
                continue
            if str(task.task_id) in protected or getattr(task, "in_service_by", None) not in (None, ""):
                continue
            candidates.append(task)

        step_now = int(getattr(self.state, "step_index", 0))
        used: set[str] = set()
        for aid, agent in sorted(self.state.agents.items()):
            if agent.kind != AgentKind.TRUCK or bool(getattr(agent, "crashed", False)):
                continue
            feasible: List[Tuple[int, float, str, DeliveryTask]] = []
            for task in candidates:
                tid = str(task.task_id)
                if tid in used or not self._truck_can_service_task(str(aid), task):
                    continue
                slack = int(max(int(task.deadline_step) - step_now, 0))
                dist = float(self._decision_shortest_path_distance(int(agent.node), int(task.demand_node)))
                feasible.append((slack, dist, tid, task))
            if not feasible:
                continue
            _, _, tid, _ = min(feasible, key=lambda item: (item[0], item[1], item[2]))
            # Remove only docked/grounded duplicate advertisements. Airborne
            # commitments were excluded above and remain execution-authoritative.
            for other_aid, gid in list(self._effective_goals.items()):
                if str(other_aid) == str(aid) or str(gid) != tid:
                    continue
                other = self.state.agents.get(str(other_aid), None)
                if other is None or other.kind != AgentKind.UAV or other.follow_target is not None:
                    self._effective_goals[str(other_aid)] = None
            self._effective_goals[str(aid)] = tid
            used.add(tid)

    def _b_route_stability_active(self) -> bool:
        """Return whether the road-only B commitment window is active.

        C already obtains commitment from the physical blackout protocol.  The
        B transfer is deliberately narrower: only the L/B (or M/B) truck route
        may receive a short hold, never the UAV sortie contract itself.
        """
        return bool(
            getattr(self.cfg, "hrl_b_route_stability_enabled", False)
            and str(getattr(self.cfg, "scenario", "")).upper() == "B"
            and not bool(getattr(self.cfg, "enable_comm_blackout", False))
        )

    def _b_route_stability_goal_valid(self, aid: str, goal_id: Optional[str]) -> bool:
        """Check that a held B truck goal is still safe to keep.

        This is intentionally conservative.  A held goal must resolve to the
        same truck's active routine task and have a finite current road path;
        claimed ownership, completion, inventory and road reachability are all
        rechecked on every planning step.
        """
        aid_s = str(aid)
        st = self.state.agents.get(aid_s)
        if st is None or bool(getattr(st, "crashed", False)):
            return False
        if goal_id is None:
            return False
        task = self.state.tasks.get(str(goal_id))
        if st.kind == AgentKind.TRUCK:
            if task is None or task.kind != TaskKind.NORMAL:
                return False
            # Only stabilize an already-claimed contract.  Holding a merely
            # serviceable pending task can mask a better first assignment on
            # B, whereas a claimed route has an explicit ownership anchor and
            # is the safe part of C's anti-churn behavior to transfer.
            task_active = bool(
                task.status == TaskStatus.CLAIMED
                and task.assigned_to is not None
                and str(task.assigned_to) == aid_s
            )
            if not task_active or st.node is None:
                return False
            try:
                route_dist = float(
                    self._decision_shortest_path_distance(
                        int(st.node), int(task.demand_node)
                    )
                )
            except Exception:
                return False
            return bool(np.isfinite(route_dist))
        # Docked UAVs may retain a live emergency contract for the same short
        # window.  Airborne UAVs are handled by the strict sortie contract and
        # never enter this branch.
        if st.kind == AgentKind.UAV and st.follow_target is not None:
            if task is None or not self._task_is_uav_delivery(task):
                return False
            if task.status not in (TaskStatus.PENDING, TaskStatus.CLAIMED):
                return False
            if task.status == TaskStatus.CLAIMED and (
                task.assigned_to is None or str(task.assigned_to) != aid_s
            ):
                return False
            return True
        return False

    def _b_route_stability_should_hold(
        self,
        aid: str,
        previous_goal: Optional[str],
        incoming_goal: Optional[str],
        now_step: int,
    ) -> bool:
        """Apply a bounded B-only anti-churn hold for routine truck routes."""
        if not self._b_route_stability_active():
            return False
        if not self._b_route_stability_goal_valid(aid, previous_goal):
            self._b_route_stability_goal_step.pop(str(aid), None)
            return False
        # A new emergency task or explicit truck/UAV support target must be
        # allowed through immediately.  Only None/normal-task replacements are
        # considered low-value churn candidates.
        st = self.state.agents.get(str(aid))
        incoming_task = self.state.tasks.get(str(incoming_goal)) if incoming_goal is not None else None
        if st is not None and st.kind == AgentKind.TRUCK:
            if incoming_task is not None and incoming_task.kind != TaskKind.NORMAL:
                return False
        elif st is not None and st.kind == AgentKind.UAV:
            if incoming_task is not None and not self._task_is_uav_delivery(incoming_task):
                return False
        else:
            return False
        if incoming_goal is not None and incoming_task is None:
            # Agent-to-agent support/recovery target (e.g. ``TC-2``) is a hard
            # coordination event, not a routine task switch.
            return False
        hold_steps = int(max(getattr(self.cfg, "hrl_b_route_stability_hold_steps", 0), 0))
        if hold_steps <= 0:
            return False
        key = str(aid)
        started = self._b_route_stability_goal_step.get(key)
        if started is None or previous_goal != self._effective_goals.get(key):
            started = int(now_step)
            self._b_route_stability_goal_step[key] = started
        return bool(int(now_step) - int(started) < hold_steps)

    def set_recommended_goals(self, goals: Dict[str, Optional[str]]) -> None:
        prev_effective = dict(self._effective_goals)
        self._recommended_goals = {
            str(k): (None if v is None else str(v)) for k, v in goals.items()
        }
        island_ids_step = self._current_island_emergency_task_ids()
        route_plan_v2_enabled = er_hlns_route_plan_active(self)
        island_override_enabled = bool(
            getattr(self.cfg, "uav_env_island_goal_override_enabled", False)
            and not route_plan_v2_enabled
        )
        strict_sortie_contract = bool(getattr(self.cfg, "uav_strict_sortie_contract_enabled", True))
        # Communication blackout freeze: blocked agents keep previous effective goal.
        for aid in self.state.agents:
            incoming = self._recommended_goals.get(str(aid), None)
            st = self.state.agents.get(str(aid), None)
            # A task-bound truck transfer is only the first leg of the same
            # delivery.  Once the UAV binds to the receiving truck, retain the
            # locally accepted task until it can safely relaunch.  This narrow
            # continuation does not alter either truck's planned route.
            post_transfer_tid = str(
                getattr(self, "_uav_post_transfer_contract_task", {}).get(
                    str(aid), ""
                )
            ).strip()
            if (
                strict_sortie_contract
                and st is not None
                and st.kind == AgentKind.UAV
                and st.follow_target is not None
                and post_transfer_tid
            ):
                post_transfer_task = self.state.tasks.get(post_transfer_tid)
                contract_tid = str(
                    self._uav_sortie_contract_task.get(str(aid), "")
                ).strip()
                if (
                    post_transfer_task is not None
                    and contract_tid == post_transfer_tid
                    and self._task_is_uav_delivery(post_transfer_task)
                    and post_transfer_task.status
                    in (TaskStatus.PENDING, TaskStatus.CLAIMED)
                    and bool(
                        self._uav_loaded_for_task(
                            str(aid), post_transfer_task
                        )
                    )
                ):
                    self._effective_goals[str(aid)] = post_transfer_tid
                    continue
                self._uav_post_transfer_contract_task.pop(str(aid), None)
            # A docked, fully reloaded UAV can safely clear a stale forced-RTH
            # latch before a new sortie. Launch feasibility is still checked
            # by _uav_launch_gate_check; this never clears an airborne or
            # low-battery recovery latch.
            if (
                isinstance(self.algorithm_profile, AlgorithmProfile)
                and self.algorithm_profile.has(
                    ER_HLNS_B_DOCKED_LATCH_REARM_CAPABILITY
                )
                and st is not None
                and st.kind == AgentKind.UAV
                and st.follow_target is not None
                and bool(self._uav_forced_rth_latch.get(str(aid), False))
                and bool(self._uav_loaded(str(aid)))
                and float(getattr(st, "battery", 0.0)) >= 0.95
                and not bool(getattr(st, "uav_needs_reload_flag", False))
                and int(getattr(st, "uav_reload_timer", 0)) <= 0
            ):
                self._uav_forced_rth_latch[str(aid)] = False
                self._uav_forced_rth_start_step.pop(str(aid), None)
            # Environment-side sortie contract.  An airborne loaded UAV may
            # only pursue the task it launched for; an empty airborne UAV may
            # only recover to a truck.  This executes before communication
            # fallback so a blackout cannot silently retarget a live sortie.
            if strict_sortie_contract and st is not None and st.kind == AgentKind.UAV and st.follow_target is None:
                # Hard recovery can temporarily outrank the accepted delivery
                # contract, but a forced-RTH latch alone must not preempt a
                # still-feasible outbound delivery leg.  Keep the contract
                # installed (so recovery and recharge do not lose the task)
                # while deciding whether a truck goal is actually required.
                force_thr = float(
                    np.clip(
                        getattr(
                            self.cfg,
                            "uav_low_battery_force_recover_threshold",
                            0.25,
                        ),
                        0.0,
                        1.0,
                    )
                )
                recovery_suspended = getattr(
                    self, "_uav_sortie_recovery_suspended", set()
                )
                contract_tid = self._uav_sortie_contract_task.get(str(aid), None)
                contract_task = self.state.tasks.get(str(contract_tid)) if contract_tid is not None else None
                forced_rth = bool(self._uav_forced_rth_latch.get(str(aid), False))
                battery = float(getattr(st, "battery", 0.0))
                hard_low_battery = bool(battery <= force_thr)
                contract_active = bool(
                    contract_task is not None
                    and self._task_is_uav_delivery(contract_task)
                    and contract_task.status in (TaskStatus.PENDING, TaskStatus.CLAIMED)
                    and bool(self._uav_loaded_for_task(str(aid), contract_task))
                )
                rendezvous_delivery_contract = bool(
                    contract_active
                    and _UAV_SORTIE_DELIVERY_LEG_RECOVERY_BYPASS_ENABLED
                    and str(
                        self._uav_last_launch_reason.get(str(aid), "")
                    ).startswith("rendezvous_safe")
                )
                if contract_active and hard_low_battery:
                    recovery_suspended.add(str(aid))
                    truck_id, _ = self._nearest_truck_from_xy(
                        self._agent_xy(str(aid))
                    )
                    self._effective_goals[str(aid)] = (
                        None if truck_id is None else str(truck_id)
                    )
                    continue
                if not rendezvous_delivery_contract:
                    # Legacy behavior remains authoritative for direct-safe,
                    # stale and energy-infeasible sorties.  This isolates the
                    # new rule to contracts whose launch gate explicitly
                    # approved a rendezvous/cross-truck recovery chain.
                    if contract_active:
                        self._effective_goals[str(aid)] = str(contract_tid)
                        continue
                    self._uav_sortie_contract_task.pop(str(aid), None)
                    self._uav_sortie_contract_version.pop(str(aid), None)
                    if not bool(self._uav_loaded(str(aid))):
                        truck_id, _ = self._nearest_truck_from_xy(
                            self._agent_xy(str(aid))
                        )
                        self._effective_goals[str(aid)] = (
                            None if truck_id is None else str(truck_id)
                        )
                        continue
                if contract_active:
                    # A forced-RTH latch may not interrupt an accepted loaded
                    # delivery when the remaining *outbound* leg is still
                    # energy-feasible.  Do not use a full round-trip gate here:
                    # recovery is evaluated after delivery (or when the hard
                    # low-battery threshold is reached).
                    delivery_leg_feasible = bool(
                        self._uav_sortie_delivery_leg_feasible(
                            str(aid), contract_task
                        )
                    )

                    hard_recovery_active = bool(
                        hard_low_battery or (forced_rth and not delivery_leg_feasible)
                    )
                    if hard_recovery_active:
                        recovery_suspended.add(str(aid))
                    else:
                        # Clear a stale suspension when a latched sortie has
                        # regained a feasible delivery leg; the latch itself
                        # is handled by the normal post-delivery recovery path.
                        recovery_suspended.discard(str(aid))
                    if str(aid) in recovery_suspended:
                        truck_id, _ = self._nearest_truck_from_xy(
                            self._agent_xy(str(aid))
                        )
                        self._effective_goals[str(aid)] = (
                            None if truck_id is None else str(truck_id)
                        )
                        continue
                    self._effective_goals[str(aid)] = str(contract_tid)
                    continue

                # Historical broader no-contract recovery-suspension branch is
                # retained but disabled.  The experiment is intentionally
                # limited to a live rendezvous contract; stale/no-contract
                # sorties keep the established fallback below.
                if False:  # legacy experiment retained for auditability
                    hard_recovery_active = bool(forced_rth or hard_low_battery)
                    if hard_recovery_active:
                        recovery_suspended.add(str(aid))
                    if str(aid) in recovery_suspended:
                        truck_id, _ = self._nearest_truck_from_xy(
                            self._agent_xy(str(aid))
                        )
                        self._effective_goals[str(aid)] = (
                            None if truck_id is None else str(truck_id)
                        )
                        continue
            # Pre-launch commitment: when a UAV is docked, loaded, and has an
            # emergency assignment, retain it while charging. This prevents a
            # replan from dropping a feasible nearby delivery merely because
            # the UAV needs one or two more charging steps. It is not a blind
            # launch permission: _uav_launch_gate_check remains authoritative.
            if (
                strict_sortie_contract
                and bool(getattr(self.cfg, "uav_docked_sortie_commitment_enabled", True))
                and st is not None
                and st.kind == AgentKind.UAV
                and st.follow_target is not None
                and bool(self._uav_loaded(str(aid)))
            ):
                # Docking completes the hard-recovery phase.  The task
                # contract remains installed and can be resumed after normal
                # recharge/relaunch checks.
                getattr(
                    self, "_uav_sortie_recovery_suspended", set()
                ).discard(str(aid))
                contract_tid = self._uav_sortie_contract_task.get(str(aid), None)
                incoming_task = self._task_by_id_if_active(incoming, aid=str(aid))
                contract_task = self._task_by_id_if_active(contract_tid, aid=str(aid))
                profile_lock_enabled = bool(
                    isinstance(self.algorithm_profile, AlgorithmProfile)
                    and self.algorithm_profile.has(
                        ER_HLNS_B_PRELAUNCH_CONTRACT_LOCK_CAPABILITY
                    )
                )
                owner_lock_enabled = bool(
                    route_plan_v2_enabled
                    and (
                        profile_lock_enabled
                        or bool(
                            getattr(
                                self.cfg,
                                "uav_docked_contract_owner_lock_enabled",
                                True,
                            )
                        )
                    )
                )
                contract_owner_matches = bool(
                    contract_task is not None
                    and str(
                        getattr(contract_task, "route_contract_owner", "")
                        or ""
                    )
                    == str(aid)
                )
                incoming_owner_matches = bool(
                    incoming_task is not None
                    and str(
                        getattr(incoming_task, "route_contract_owner", "")
                        or ""
                    )
                    == str(aid)
                )
                installed_version = int(
                    max(
                        self._uav_sortie_contract_version.get(str(aid), 0),
                        0,
                    )
                )
                current_contract_version = int(
                    max(
                        getattr(contract_task, "route_contract_version", 0)
                        if contract_task is not None
                        else 0,
                        0,
                    )
                )
                contract_version_matches = bool(
                    installed_version > 0
                    and installed_version == current_contract_version
                )
                # Reconcile a stale docked contract only through the atomic
                # owner/version published by layer 1. An airborne contract is
                # handled above and remains execution-authoritative.
                if (
                    contract_task is not None
                    and owner_lock_enabled
                    and (
                        not contract_owner_matches
                        or not contract_version_matches
                    )
                ):
                    self._uav_sortie_contract_task.pop(str(aid), None)
                    self._uav_sortie_contract_version.pop(str(aid), None)
                    contract_task = None
                if (
                    incoming_task is not None
                    and self._task_is_uav_delivery(incoming_task)
                    and (not owner_lock_enabled or incoming_owner_matches)
                    and contract_task is None
                ):
                    contract_task = incoming_task
                    self._uav_sortie_contract_task[str(aid)] = str(incoming_task.task_id)
                    self._uav_sortie_contract_version[str(aid)] = int(
                        max(
                            getattr(incoming_task, "route_contract_version", 0),
                            0,
                        )
                    )
                    contract_owner_matches = bool(incoming_owner_matches)
                # Full charge is not a release event. Keep a loaded docked
                # UAV on its current owned task until launch or an atomic
                # owner/version change. The launch gate still decides whether
                # takeoff is safe, so this cannot force an infeasible sortie.
                if (
                    contract_task is not None
                    and self._task_is_uav_delivery(contract_task)
                    and (
                        (owner_lock_enabled and contract_owner_matches)
                        or (
                            not owner_lock_enabled
                            and float(getattr(st, "battery", 0.0))
                            < 1.0 - 1e-9
                        )
                    )
                ):
                    self._effective_goals[str(aid)] = str(contract_task.task_id)
                    continue
            if bool(self.comm_blocked.get(aid, False)):
                if aid not in self._effective_goals:
                    self._effective_goals[aid] = incoming
                continue

            effective_goal = incoming
            if st is not None and st.kind == AgentKind.UAV and st.follow_target is not None:
                eff_task = self._task_by_id_if_active(effective_goal, aid=aid)
                if eff_task is not None and self._task_is_uav_delivery(eff_task):
                    keep_task_goal = bool(getattr(self.cfg, "uav_docked_keep_task_goal_enabled", True))
                    if (not keep_task_goal) and (not bool(self._uav_docked_task_actionable_now(str(aid), eff_task))):
                        effective_goal = str(st.follow_target)
                elif effective_goal is None:
                    effective_goal = str(st.follow_target)
            if (
                island_override_enabled
                and st is not None
                and st.kind == AgentKind.UAV
                and bool(island_ids_step)
                and (
                    (not self._uav_recovery_required(str(aid)))
                    or (
                        st.follow_target is not None
                        and bool(self._uav_loaded(str(aid)))
                    )
                )
            ):
                rec_task = self._task_by_id_if_active(effective_goal, aid=aid)
                rec_island = bool(
                    rec_task is not None
                    and rec_task.kind == TaskKind.EMERGENCY
                    and str(rec_task.task_id) in island_ids_step
                )
                if not rec_island:
                    best_tid: Optional[str] = None
                    best_dist = float("inf")
                    best_launchable = -1
                    for tid in sorted(island_ids_step):
                        t = self._task_by_id_if_active(str(tid), aid=aid)
                        if t is None or t.kind != TaskKind.EMERGENCY:
                            continue
                        d = float(self._agent_distance_to_task(str(aid), t))
                        launchable = 0
                        # When UAV is docked, only prioritize island goals that pass
                        # launch gate; this avoids futile far-island overrides.
                        if st.follow_target is not None:
                            prev_goal_eval = self._effective_goals.get(str(aid), None)
                            try:
                                self._effective_goals[str(aid)] = str(t.task_id)
                                if bool(self._uav_docked_task_actionable_now(str(aid), t)):
                                    launchable = 1
                            except Exception:
                                launchable = 0
                            finally:
                                self._effective_goals[str(aid)] = prev_goal_eval

                        better = False
                        if launchable > best_launchable:
                            better = True
                        elif launchable == best_launchable and d < best_dist:
                            better = True
                        if better:
                            best_launchable = int(launchable)
                            best_dist = float(d)
                            best_tid = str(t.task_id)

                    if best_tid is not None:
                        if st.follow_target is None or best_launchable > 0:
                            # Island-priority override: when emergency island tasks exist,
                            # UAV should prefer nearest launchable (or nearest airborne) island target.
                            effective_goal = str(best_tid)

            # B-only anti-churn transfer: C's blackout protocol naturally keeps
            # the last effective route for a short period.  Reproduce only that
            # useful part for truck routine routes, and only while the old task
            # is still owned, serviceable and reachable.  Explicit support or
            # emergency goals bypass the hold immediately.
            previous_goal = prev_effective.get(str(aid), None)
            if (
                st is not None
                and st.kind in (AgentKind.TRUCK, AgentKind.UAV)
                and (st.kind != AgentKind.UAV or st.follow_target is not None)
                and previous_goal != effective_goal
                and self._b_route_stability_should_hold(
                    str(aid),
                    previous_goal,
                    effective_goal,
                    int(getattr(self.state, "step_index", 0)),
                )
            ):
                effective_goal = previous_goal
            elif previous_goal != effective_goal:
                self._b_route_stability_goal_step.pop(str(aid), None)

            self._effective_goals[aid] = effective_goal

        # Docked-UAV periodic nearest-task re-target with distinct assignment.
        # This is optional and disabled in paper mainline by default.
        if not route_plan_v2_enabled:
            self._assign_distinct_docked_uav_emergency_goals()

        # Service-eligibility sensitivity: a shared execution-layer fallback
        # may dispatch road-feasible trucks to near-deadline TC tasks.
        self._assign_conditional_truck_emergency_goals()

        # Final exclusivity guard: at most one agent may hold one pending task goal.
        self._enforce_unique_pending_task_goals()

        changed = 0
        assigned_total = 0
        assigned_truck = 0
        assigned_uav = 0
        for aid, st in self.state.agents.items():
            gid = self._effective_goals.get(str(aid), None)
            if prev_effective.get(str(aid), None) != gid:
                changed += 1
            if gid is not None:
                assigned_total += 1
                if st.kind == AgentKind.TRUCK:
                    assigned_truck += 1
                elif st.kind == AgentKind.UAV:
                    assigned_uav += 1
        self.triggered_replans_step = int(changed)
        self.triggered_replans_total += int(changed)
        self.last_assignment_summary = {
            "assigned_total": int(assigned_total),
            "assigned_truck": int(assigned_truck),
            "assigned_uav": int(assigned_uav),
        }

    def note_planner_replan(self, refresh_flags: Optional[Dict[str, bool]] = None, reason: str = "none") -> None:
        flags = dict(refresh_flags or {})
        did_refresh = bool(flags.get("refresh", True))
        by_map_update = bool(flags.get("map_update", False))
        self.planner_refresh_map_update_step = bool(did_refresh and by_map_update)
        self.planner_last_replan_reason = str(reason)
        if self.planner_refresh_map_update_step:
            self.planner_replan_due_to_new_road_info_count_total += 1

    def _has_active_assigned_task(self, aid: str) -> bool:
        for t in self.state.tasks.values():
            if (
                t.status in (TaskStatus.PENDING, TaskStatus.CLAIMED)
                and t.assigned_to is not None
                and str(t.assigned_to) == str(aid)
            ):
                return True
        return False

    def _assigned_task(self, aid: str) -> Optional[DeliveryTask]:
        agent_kind = self.state.agents[str(aid)].kind
        chosen: Optional[DeliveryTask] = None
        for t in self.state.tasks.values():
            if (
                t.status in (TaskStatus.PENDING, TaskStatus.CLAIMED)
                and t.assigned_to is not None
                and str(t.assigned_to) == str(aid)
            ):
                if agent_kind == AgentKind.UAV and not self._task_is_uav_delivery(t):
                    continue
                if chosen is None or t.deadline_step < chosen.deadline_step:
                    chosen = t
        return chosen

    def _agent_task_distance_norm(self, aid: str) -> float:
        t = self._pbrs_target_task(aid)
        if t is None:
            return 1.0
        a = self.state.agents[aid]
        node = self.topology.nodes[int(t.demand_node)]
        if a.pos_xy is not None:
            d = float(np.hypot(a.pos_xy[0] - node.x, a.pos_xy[1] - node.y))
        else:
            cur = self.topology.nodes[int(a.node or 0)]
            d = float(np.hypot(cur.x - node.x, cur.y - node.y))
        return float(np.clip(d / 3000.0, 0.0, 1.0))

    def _agent_xy(self, aid: str) -> Tuple[float, float]:
        a = self.state.agents[aid]
        if a.pos_xy is not None:
            return float(a.pos_xy[0]), float(a.pos_xy[1])
        return self._node_xy(int(a.node or 0))

    def _agent_task_rel(self, aid: str, task: DeliveryTask) -> Tuple[float, float, float]:
        ax, ay = self._agent_xy(aid)
        tn = self.topology.nodes[int(task.demand_node)]
        dx = float(tn.x - ax)
        dy = float(tn.y - ay)
        dist = float(np.hypot(dx, dy))
        return (
            float(np.clip(dx / 3000.0, -1.0, 1.0)),
            float(np.clip(dy / 3000.0, -1.0, 1.0)),
            float(np.clip(dist / 3000.0, 0.0, 1.0)),
        )

    def _agent_distance_to_task(self, aid: str, task: DeliveryTask) -> float:
        a = self.state.agents[aid]
        node = self.topology.nodes[int(task.demand_node)]
        if a.kind == AgentKind.TRUCK and a.node is not None:
            g = self._decision_shortest_path_distance(
                int(a.node), int(task.demand_node)
            )
            if np.isfinite(g):
                return float(g)
            cur = self.topology.nodes[int(a.node or 0)]
            return float(np.hypot(cur.x - node.x, cur.y - node.y))
        if a.pos_xy is not None:
            d = float(np.hypot(a.pos_xy[0] - node.x, a.pos_xy[1] - node.y))
        else:
            cur = self.topology.nodes[int(a.node or 0)]
            d = float(np.hypot(cur.x - node.x, cur.y - node.y))
        return d

    def _task_visible_to_agent(self, aid: str, task: DeliveryTask) -> bool:
        s = self.state.agents[str(aid)]
        if s.kind == AgentKind.UAV and not self._task_is_uav_delivery(task):
            return False
        if task.status == TaskStatus.PENDING:
            return bool(self.is_task_serviceable_by_agent(str(aid), task))
        if task.status == TaskStatus.CLAIMED and task.assigned_to is not None:
            return str(task.assigned_to) == str(aid)
        return False

    def _task_by_id_if_active(
        self, task_id: Optional[str], aid: Optional[str] = None
    ) -> Optional[DeliveryTask]:
        if task_id is None:
            return None
        t = self.state.tasks.get(str(task_id))
        if t is None:
            return None
        if aid is not None:
            a = self.state.agents.get(str(aid))
            if a is not None and a.kind == AgentKind.UAV and not self._task_is_uav_delivery(t):
                return None
        if t.status == TaskStatus.PENDING:
            if aid is None:
                return t
            return t if self.is_task_serviceable_by_agent(str(aid), t) else None
        if aid is not None and t.status == TaskStatus.CLAIMED:
            if t.assigned_to is not None and str(t.assigned_to) == str(aid):
                return t
            return None
        return None

    def _pbrs_target_task(self, aid: str) -> Optional[DeliveryTask]:
        # HRL-authoritative target: only explicit recommendation.
        rec_tid = self._effective_goals.get(
            str(aid), self._recommended_goals.get(str(aid), None)
        )
        rec_task = self._task_by_id_if_active(rec_tid, aid=aid)
        if rec_task is not None:
            return rec_task
        return self._assigned_task(aid)

    def _service_rounds(self, aid: str, task: DeliveryTask) -> int:
        s = self.state.agents[aid]
        if s.kind == AgentKind.UAV:
            return int(max(1, int(self.cfg.unload_rounds_uav)))
        if self._task_supply_type(task) == "emergency":
            return int(max(1, int(self.cfg.unload_rounds_emergency)))
        return int(max(1, int(self.cfg.unload_rounds_normal)))

    def _servicing_agents(self) -> set:
        out = set()
        for t in self.state.tasks.values():
            if (
                t.status == TaskStatus.CLAIMED
                and t.in_service_by is not None
                and int(t.service_remaining) > 0
            ):
                out.add(str(t.in_service_by))
                if self._task_is_bulk_relay(t):
                    out.update(str(aid) for aid in getattr(t, "relay_service_agents", ()) if str(aid))
        return out

    @staticmethod
    def _point_segment_distance(
        px: float, py: float, ax: float, ay: float, bx: float, by: float
    ) -> float:
        vx = float(bx - ax)
        vy = float(by - ay)
        wx = float(px - ax)
        wy = float(py - ay)
        vv = float(vx * vx + vy * vy)
        if vv <= 1e-12:
            return float(np.hypot(px - ax, py - ay))
        t = float(np.clip((wx * vx + wy * vy) / vv, 0.0, 1.0))
        cx = float(ax + t * vx)
        cy = float(ay + t * vy)
        return float(np.hypot(px - cx, py - cy))

    def _decision_mode_shared(self) -> bool:
        mode = str(getattr(self.cfg, "road_awareness_mode", "perfect")).strip().lower()
        return bool(getattr(self.cfg, "road_shared_awareness_enabled", True) and mode == "shared")

    def _decision_cache_token(self) -> Tuple[int, int, int, int, int, int]:
        shared_mode = bool(self._decision_mode_shared())
        risk_aware = bool(getattr(self.cfg, "road_risk_aware_routing_enabled", True))
        blocked_edges = (
            set(self._shared_known_blocked_edges)
            if shared_mode
            else set(self.topology.blocked_edges)
        )
        # Count alone is insufficient when one edge reopens while another is
        # blocked in the same update.  The fingerprint changes only when the
        # actual known blocked-edge set changes.
        blocked_version = int(
            hash(
                tuple(
                    sorted(
                        (min(int(a), int(b)), max(int(a), int(b)))
                        for a, b in blocked_edges
                    )
                )
            )
        )
        road_version_only = bool(getattr(self.cfg, "decision_sp_cache_road_version_only", False))
        return (
            # Exact mode retains the old per-step invalidation because edge-risk
            # costs are refreshed each step.  The optional diagnostic mode keeps
            # route distances until the known road graph actually changes.
            0 if road_version_only else int(getattr(self.state, "step_index", 0)),
            int(getattr(self, "_shared_map_update_count_total", 0)),
            int(getattr(self, "_unknown_blocked_edge_hit_total", 0)),
            int(blocked_version),
            int(1 if shared_mode else 0),
            int(1 if risk_aware else 0),
        )

    def _ensure_decision_runtime_caches(self) -> Tuple[int, int, int, int, int, int]:
        token = self._decision_cache_token()
        if getattr(self, "_decision_runtime_cache_token", None) != token:
            self._decision_runtime_cache_token = token
            self._decision_blocked_edges_cache = None
            self._decision_is_blocked_cache = {}
            self._decision_edge_cost_cache = {}
            self._decision_sp_cache = {}
        return token

    def _init_shared_road_awareness_state(self) -> None:
        self._shared_map_update_event_step = False
        self._shared_map_update_count_total = 0
        self._shared_map_new_blocked_step = 0
        self._shared_map_new_blocked_total = 0
        self._shared_map_cleared_step = 0
        self._shared_map_cleared_total = 0
        self._shared_discovery_uav_step = 0
        self._shared_discovery_uav_total = 0
        self._shared_discovery_truck_step = 0
        self._shared_discovery_truck_total = 0
        self._unknown_blocked_edge_hit_step = 0
        self._unknown_blocked_edge_hit_total = 0
        self._shared_last_update_reason = "none"

        if self._decision_mode_shared():
            # Shared-belief mode starts from partial cognition and is updated by scout/contact.
            self._shared_known_blocked_edges = set()
        else:
            # Legacy perfect-awareness mode mirrors physical blocked graph.
            self._shared_known_blocked_edges = set(self.topology.blocked_edges)

    def _decision_blocked_edges(self) -> set:
        self._ensure_decision_runtime_caches()
        cached = getattr(self, "_decision_blocked_edges_cache", None)
        if cached is not None:
            return cached
        if self._decision_mode_shared():
            cached = set(self._shared_known_blocked_edges)
        else:
            cached = set(self.topology.blocked_edges)
        self._decision_blocked_edges_cache = cached
        return cached

    def _decision_is_blocked(self, src: int, dst: int) -> bool:
        self._ensure_decision_runtime_caches()
        k = (min(int(src), int(dst)), max(int(src), int(dst)))
        cached = self._decision_is_blocked_cache.get(k, None)
        if cached is not None:
            return bool(cached)
        out = bool(k in self._decision_blocked_edges())
        self._decision_is_blocked_cache[k] = bool(out)
        return bool(out)

    def _decision_neighbors(self, node_id: int) -> List[int]:
        node = int(node_id)
        blocked_edges = self._decision_blocked_edges()
        out: List[int] = []
        for nb in self.topology.adjacency.get(node, set()):
            k = (min(int(node), int(nb)), max(int(node), int(nb)))
            if k not in blocked_edges:
                out.append(int(nb))
        return out

    def _decision_blocked_ratio(self) -> float:
        total = sum(len(v) for v in self.topology.adjacency.values()) // 2
        if total <= 0:
            return 0.0
        return float(len(self._decision_blocked_edges()) / float(total))

    def _decision_edge_cost(self, src: int, dst: int) -> float:
        self._ensure_decision_runtime_caches()
        edge_key = (min(int(src), int(dst)), max(int(src), int(dst)))
        cached = self._decision_edge_cost_cache.get(edge_key, None)
        if cached is not None:
            return float(cached)

        base = float(self.topology.edge_distance(int(src), int(dst)))
        if not bool(getattr(self.cfg, "road_risk_aware_routing_enabled", True)):
            self._decision_edge_cost_cache[edge_key] = float(base)
            return float(base)

        p_edge = 0.0
        if hasattr(self, "hazards") and self.hazards is not None:
            p_edge = float(
                np.clip(
                    getattr(self.hazards, "last_edge_pstep", {}).get(
                        edge_key, getattr(self.hazards, "last_pstep_mean", 0.0)
                    ),
                    0.0,
                    1.0,
                )
            )

        v_base = float(
            np.clip(self.topology.edge_attr(int(src), int(dst)).base_vulnerability, 0.0, 1.0)
        )
        prob_w = float(max(getattr(self.cfg, "road_risk_edge_prob_weight", 1.25), 0.0))
        vuln_w = float(max(getattr(self.cfg, "road_risk_vulnerability_weight", 0.35), 0.0))
        cap = float(max(getattr(self.cfg, "road_risk_cost_multiplier_cap", 3.0), 1.0))

        risk_score = float(np.clip(p_edge + vuln_w * v_base, 0.0, 1.0))
        mult = float(np.clip(1.0 + prob_w * risk_score, 1.0, cap))
        out = float(base * mult)
        self._decision_edge_cost_cache[edge_key] = float(out)
        return float(out)

    def _decision_shortest_path_distance(self, src: int, dst: int) -> float:
        src_i = int(src)
        dst_i = int(dst)
        if src_i == dst_i:
            return 0.0

        token = self._ensure_decision_runtime_caches()
        shared_mode = bool(token[4])
        risk_aware = bool(token[5])

        k = (min(src_i, dst_i), max(src_i, dst_i))
        cached = self._decision_sp_cache.get(k, None)
        if cached is not None:
            return float(cached)

        if (not shared_mode) and (not risk_aware):
            d = float(self.topology.shortest_path_distance(src_i, dst_i, ignore_blocked=False))
            self._decision_sp_cache[k] = float(d)
            return float(d)

        # One single-source Dijkstra supplies all candidate destinations from
        # this source.  Later pair queries reuse the filled cache instead of
        # repeating a graph search for every task/anchor candidate.
        inf = float("inf")
        dist = {int(n): inf for n in self.topology.nodes.keys()}
        dist[src_i] = 0.0
        heap: list[tuple[float, int]] = [(0.0, src_i)]
        while heap:
            cur_d, cur = heapq.heappop(heap)
            if cur_d > float(dist[int(cur)]) + 1e-12:
                continue
            for nb in self.topology.adjacency.get(int(cur), set()):
                if self._decision_is_blocked(int(cur), int(nb)):
                    continue
                nd = float(cur_d + self._decision_edge_cost(int(cur), int(nb)))
                if nd + 1e-12 < float(dist[int(nb)]):
                    dist[int(nb)] = float(nd)
                    heapq.heappush(heap, (float(nd), int(nb)))

        for node, value in dist.items():
            if int(node) == src_i:
                continue
            self._decision_sp_cache[(min(src_i, int(node)), max(src_i, int(node)))] = float(value)
        return float(dist.get(dst_i, inf))

    def _current_island_emergency_task_ids(self) -> set:        # Island task: emergency task unreachable from all trucks in current decision graph.
        # Cache is step-tokenized to avoid repeated shortest-path scans within one step.
        token = (
            int(self.state.step_index),
            int(getattr(self, "_shared_map_update_count_total", 0)),
            int(getattr(self, "_unknown_blocked_edge_hit_total", 0)),
            int(len(self._decision_blocked_edges())),
        )
        if getattr(self, "_cached_island_task_ids_token", None) == token:
            return set(getattr(self, "_cached_island_task_ids", set()))

        truck_nodes = [
            int(ts.node)
            for ts in self.state.agents.values()
            if ts.kind == AgentKind.TRUCK and ts.node is not None and (not bool(getattr(ts, "crashed", False)))
        ]
        if not truck_nodes:
            self._cached_island_task_ids_token = token
            self._cached_island_task_ids = set()
            return set()

        out = set()
        for t in self.state.tasks.values():
            if t.kind != TaskKind.EMERGENCY:
                continue
            if t.status not in (TaskStatus.PENDING, TaskStatus.CLAIMED):
                continue
            tid = str(t.task_id)
            node = int(t.demand_node)
            # Forced-island tasks are treated as island ground-demand by design,
            # independent of shared-map discovery latency.
            if tid in self._forced_island_task_ids:
                out.add(tid)
                continue
            reachable = False
            for tn in truck_nodes:
                d = float(self._decision_shortest_path_distance(int(tn), int(node)))
                if np.isfinite(d):
                    reachable = True
                    break
            if not reachable:
                out.add(tid)

        self._cached_island_task_ids_token = token
        self._cached_island_task_ids = set(out)
        return out

    def _visible_edge_keys_within_radius(self, px: float, py: float, radius_m: float) -> List[Tuple[int, int]]:
        if float(radius_m) <= 0.0:
            return []
        visible: List[Tuple[int, int]] = []
        for src, nbs in self.topology.adjacency.items():
            for dst in nbs:
                if int(src) >= int(dst):
                    continue
                a = self.topology.nodes[int(src)]
                b = self.topology.nodes[int(dst)]
                d = self._point_segment_distance(float(px), float(py), a.x, a.y, b.x, b.y)
                if d <= float(radius_m):
                    visible.append((int(src), int(dst)))
        return visible

    def _update_shared_map_edge_observation(self, edge_key: Tuple[int, int], blocked_phys: bool, source: str) -> bool:
        changed = False
        key = (min(int(edge_key[0]), int(edge_key[1])), max(int(edge_key[0]), int(edge_key[1])))
        if bool(blocked_phys):
            if key not in self._shared_known_blocked_edges:
                self._shared_known_blocked_edges.add(key)
                self._shared_map_new_blocked_step += 1
                self._shared_map_new_blocked_total += 1
                changed = True
        else:
            if key in self._shared_known_blocked_edges:
                self._shared_known_blocked_edges.discard(key)
                self._shared_map_cleared_step += 1
                self._shared_map_cleared_total += 1
                changed = True

        if changed:
            if source == "uav":
                self._shared_discovery_uav_step += 1
                self._shared_discovery_uav_total += 1
            elif source.startswith("truck"):
                self._shared_discovery_truck_step += 1
                self._shared_discovery_truck_total += 1
            self._shared_map_update_event_step = True
            self._shared_map_update_count_total += 1
            self._shared_last_update_reason = str(source)
        return changed

    def _update_shared_map_from_scout(self, aid: str, radius_m: float, source: str) -> int:
        if (not self._decision_mode_shared()) or float(radius_m) <= 0.0:
            return 0
        s = self.state.agents.get(str(aid), None)
        if s is None or bool(getattr(s, "crashed", False)):
            return 0
        xy = s.pos_xy if s.pos_xy is not None else self._node_xy(int(s.node or 0))
        edges = self._visible_edge_keys_within_radius(float(xy[0]), float(xy[1]), float(radius_m))
        changed = 0
        for edge in edges:
            blocked_phys = bool(self.topology.is_blocked(int(edge[0]), int(edge[1])))
            if self._update_shared_map_edge_observation(edge, blocked_phys, source=source):
                changed += 1
        return int(changed)

    def _shared_awareness_step_reset(self) -> None:
        self._shared_map_update_event_step = False
        self._shared_map_new_blocked_step = 0
        self._shared_map_cleared_step = 0
        self._shared_discovery_uav_step = 0
        self._shared_discovery_truck_step = 0
        self._unknown_blocked_edge_hit_step = 0
        self._shared_last_update_reason = "none"
        self._diag_uav_task_reject_seen_step = set()
        self._diag_truck_emergency_relief_seen_step = set()
        self._diag_truck_emergency_serviceability_seen_step = set()

    def _uav_scouting_enabled(self) -> bool:
        return bool(
            getattr(self.cfg, "road_uav_scout_enabled", True)
            and uav_scout_information_active(self)
        )

    def _shared_awareness_step_update(self) -> None:
        if not self._decision_mode_shared():
            self._shared_known_blocked_edges = set(self.topology.blocked_edges)
            return

        if bool(getattr(self.cfg, "road_truck_scout_enabled", True)):
            tr = float(max(getattr(self.cfg, "road_truck_scout_radius_m", 220.0), 0.0))
            for aid, s in self.state.agents.items():
                if s.kind != AgentKind.TRUCK:
                    continue
                self._update_shared_map_from_scout(str(aid), tr, source="truck_scout")

        if self._uav_scouting_enabled():
            ur = float(max(getattr(self.cfg, "road_uav_scout_radius_m", self.cfg.uav_monitor_radius_m), 0.0))
            for aid, s in self.state.agents.items():
                if s.kind != AgentKind.UAV:
                    continue
                self._update_shared_map_from_scout(str(aid), ur, source="uav")

    def _uav_visible_edge_ratio(self, aid: str, radius_m: float) -> float:
        s = self.state.agents[aid]
        if s.pos_xy is None:
            return 0.0
        px, py = float(s.pos_xy[0]), float(s.pos_xy[1])
        total = 0
        valid = 0
        for src, nbs in self.topology.adjacency.items():
            for dst in nbs:
                if int(src) >= int(dst):
                    continue
                a = self.topology.nodes[int(src)]
                b = self.topology.nodes[int(dst)]
                d = self._point_segment_distance(px, py, a.x, a.y, b.x, b.y)
                if d <= float(radius_m):
                    total += 1
                    if not self.topology.is_blocked(int(src), int(dst)):
                        valid += 1
        if total <= 0:
            return 0.0
        return float(valid / total)

    def _uav_visible_blocked_edges(self, aid: str, radius_m: float) -> List[Tuple[int, int]]:
        s = self.state.agents[aid]
        if s.pos_xy is None or s.crashed:
            return []
        if s.follow_target is not None:
            return []
        px, py = float(s.pos_xy[0]), float(s.pos_xy[1])
        visible: List[Tuple[int, int]] = []
        for (ea, eb) in self.topology.blocked_edges:
            a = self.topology.nodes[int(ea)]
            b = self.topology.nodes[int(eb)]
            d = self._point_segment_distance(px, py, a.x, a.y, b.x, b.y)
            if d <= float(radius_m):
                k = (min(int(ea), int(eb)), max(int(ea), int(eb)))
                visible.append(k)
        return visible

    def _uav_all_settled_for_termination(self) -> bool:
        for s in self.state.agents.values():
            if s.kind != AgentKind.UAV:
                continue
            if s.crashed:
                continue
            if s.follow_target is not None:
                continue
            return False
        return True

    def _start_service_for_arrived_agents(self) -> List[Tuple[str, DeliveryTask]]:
        started: List[Tuple[str, DeliveryTask]] = []
        busy = self._servicing_agents()

        # 0) Consume previous-step monitor-radius snap intents (next-round activation).
        snap_enabled = bool(getattr(self.cfg, "uav_monitor_snap_enabled", False)) or bool(
            getattr(self.cfg, "enable_monitor_snap", False)
        )
        if (not snap_enabled) and self._uav_emergency_snap_pending:
            # Safety: if snap rule is disabled, drop stale pending intents.
            self._uav_emergency_snap_pending = {}
        if snap_enabled and self._uav_emergency_snap_pending:
            pending_items = list(self._uav_emergency_snap_pending.items())
            self._uav_emergency_snap_pending = {}
            for tid, aid in pending_items:
                t = self.state.tasks.get(str(tid))
                s = self.state.agents.get(str(aid))
                if t is None or s is None:
                    continue
                if s.kind != AgentKind.UAV or s.crashed or str(aid) in busy:
                    continue
                if t.status != TaskStatus.PENDING or not self._task_is_uav_delivery(t):
                    continue
                # UAV can only start emergency service when loaded under material gate.
                if not self._uav_can_service_task(str(aid), t):
                    continue
                v2_service = v2_authorize_service_start(self, str(aid), t)
                if v2_service is not None and not bool(v2_service[0]):
                    continue
                n = self.topology.nodes[int(t.demand_node)]
                # Snap UAV to demand node and start unloading service.
                s.pos_xy = (float(n.x), float(n.y))
                s.node = int(t.demand_node)
                s.vel_xy = (0.0, 0.0)
                t.status = TaskStatus.CLAIMED
                t.assigned_to = str(aid)
                t.in_service_by = str(aid)
                t.service_remaining = int(self._service_rounds(str(aid), t))
                record_v2_service_start(self, str(aid), t)
                started.append((str(aid), t))
                busy.add(str(aid))

        # 1) Regular arrival-based service trigger.
        for aid, s in self.state.agents.items():
            if aid in busy or s.crashed:
                continue
            if s.kind == AgentKind.TRUCK:
                if s.transit is not None or s.node is None:
                    continue
                route_v2_enabled = er_hlns_route_plan_active(self)
                route_v2_goal = (
                    self._effective_goals.get(str(aid), None)
                    if route_v2_enabled
                    else None
                )
                route_v2_assist_active = bool(
                    str(aid)
                    in dict(
                        getattr(
                            self,
                            "_planner_truck_assist_waypoint_by_truck",
                            {},
                        )
                    )
                )

                def route_v2_local_service_allowed(t: DeliveryTask) -> bool:
                    if not route_v2_enabled:
                        return True
                    if route_v2_goal is not None and str(t.task_id) == str(route_v2_goal):
                        return True
                    if (
                        t.kind == TaskKind.NORMAL
                        and not self._task_is_bulk_relay(t)
                        and not route_v2_assist_active
                    ):
                        contract_truck = str(
                            getattr(t, "route_contract_truck", "") or ""
                        )
                        # A truck physically at a direct routine task must not
                        # lose the service opportunity because a road-version
                        # replan cleared its goal during the final transit.
                        return bool(
                            route_v2_goal is None
                            or contract_truck == str(aid)
                        )
                    return False

                local_pending = [
                    t
                    for t in self.state.tasks.values()
                    if (
                        t.status == TaskStatus.PENDING
                        and int(t.demand_node) == int(s.node)
                        and route_v2_local_service_allowed(t)
                    )
                ]
                if (not local_pending) or (not any(self.is_task_serviceable_by_agent(aid, t) for t in local_pending)):
                    for t in local_pending:
                        if not self.is_task_serviceable_by_agent(aid, t):
                            self._flag_supply_block(t)
                    continue
                cands = [
                    t
                    for t in self.state.tasks.values()
                    if (
                        t.status == TaskStatus.PENDING
                        and int(t.demand_node) == int(s.node)
                        and route_v2_local_service_allowed(t)
                        and self.is_task_serviceable_by_agent(aid, t)
                    )
                ]
                if not cands:
                    continue
                target_tid = None
                target_task = self._pbrs_target_task(aid)
                if target_task is not None:
                    target_tid = str(target_task.task_id)
                cands.sort(
                    key=lambda t: (
                        0 if (target_tid is not None and str(t.task_id) == target_tid) else 1,
                        0 if t.kind == TaskKind.EMERGENCY else 1,
                        t.deadline_step,
                    )
                )
                t = cands[0]
                v2_service = v2_authorize_service_start(self, str(aid), t)
                if v2_service is not None and not bool(v2_service[0]):
                    continue
                t.status = TaskStatus.CLAIMED
                t.assigned_to = str(aid)
                t.in_service_by = str(aid)
                t.service_remaining = int(self._service_rounds(aid, t))
                record_v2_service_start(self, str(aid), t)
                started.append((aid, t))
                busy.add(str(aid))
                continue

            # UAV logic
            if s.pos_xy is None:
                continue
            if not self._uav_loaded(aid):
                continue
            immediate_cands: List[DeliveryTask] = []
            monitor_cands: List[Tuple[float, DeliveryTask]] = []
            base_delivery_radius = float(max(getattr(self.cfg, "uav_delivery_radius_m", 0.0), 0.0))
            capture_motion_factor = float(max(getattr(self.cfg, "uav_delivery_capture_motion_factor", 0.80), 0.0))
            motion_capture_radius = float(capture_motion_factor * max(float(getattr(self.cfg, "uav_max_speed_mps", 0.0)), 0.0) * max(self._dt_seconds, 0.0))
            capture_radius = float(max(base_delivery_radius, motion_capture_radius))
            route_v2_enabled = er_hlns_route_plan_active(self)
            route_v2_goal = (
                self._effective_goals.get(str(aid), None)
                if route_v2_enabled
                else None
            )
            for t in self.state.tasks.values():
                relay_join = bool(
                    self._task_is_bulk_relay(t)
                    and t.status == TaskStatus.CLAIMED
                    and str(aid) in tuple(str(uid) for uid in getattr(t, "route_contract_uav_ids", ()))
                    and str(aid) not in tuple(str(uid) for uid in getattr(t, "relay_service_agents", ()))
                )
                if (t.status != TaskStatus.PENDING and not relay_join) or not self._task_is_uav_delivery(t):
                    continue
                if route_v2_enabled and (
                    route_v2_goal is None
                    or str(t.task_id) != str(route_v2_goal)
                ):
                    continue
                if not self._uav_can_service_task(str(aid), t):
                    continue
                n = self.topology.nodes[int(t.demand_node)]
                d = float(np.hypot(float(s.pos_xy[0]) - n.x, float(s.pos_xy[1]) - n.y))
                if d <= capture_radius:
                    immediate_cands.append(t)
                elif snap_enabled and d <= float(
                    self.cfg.uav_monitor_radius_m
                ):
                    # Queue for next-round snap trigger (optional rule takeover).
                    monitor_cands.append((d, t))

            if immediate_cands:
                target_tid = None
                target_task = self._pbrs_target_task(aid)
                if target_task is not None:
                    target_tid = str(target_task.task_id)
                immediate_cands.sort(
                    key=lambda t: (
                        0 if (target_tid is not None and str(t.task_id) == target_tid) else 1,
                        t.deadline_step,
                    )
                )
                t = immediate_cands[0]
                v2_service = v2_authorize_service_start(self, str(aid), t)
                if v2_service is not None and not bool(v2_service[0]):
                    continue
                s.vel_xy = (0.0, 0.0)
                if self._task_is_bulk_relay(t) and t.status == TaskStatus.CLAIMED:
                    t.relay_service_agents = tuple(list(getattr(t, "relay_service_agents", ())) + [str(aid)])
                else:
                    t.status = TaskStatus.CLAIMED
                    t.assigned_to = str(aid)
                    t.in_service_by = str(aid)
                    t.relay_service_agents = (str(aid),) if self._task_is_bulk_relay(t) else ()
                    t.service_remaining = int(self._service_rounds(aid, t))
                record_v2_service_start(self, str(aid), t)
                started.append((aid, t))
                busy.add(str(aid))
                continue

            if monitor_cands:
                monitor_cands.sort(key=lambda x: (x[0], x[1].deadline_step))
                _, t = monitor_cands[0]
                tid = str(t.task_id)
                # keep first reservation; avoid overwriting by later UAVs
                if tid not in self._uav_emergency_snap_pending and t.assigned_to is None:
                    self._uav_emergency_snap_pending[tid] = str(aid)

        return started

    def _capture_ready_routine_tasks_before_motion(self) -> List[Tuple[str, DeliveryTask]]:
        """Atomically capture direct bulk service opportunities before truck motion.

        This is a common physical service rule for every algorithm: a stocked
        truck already at a pending direct-routine node starts unloading before
        any planner action can move it away.  ER-HLNS additionally synchronizes
        its route contract; comparison methods receive the same service event.
        """
        route_v2_active = bool(er_hlns_route_plan_active(self))
        started: List[Tuple[str, DeliveryTask]] = []
        busy = self._servicing_agents()
        for aid, s in self.state.agents.items():
            if s.kind != AgentKind.TRUCK or s.crashed or str(aid) in busy:
                continue
            # A truck in a road-edge transit state is not physically at the node.
            if s.transit is not None or s.node is None:
                continue

            onsite = [
                t
                for t in self.state.tasks.values()
                if (
                    t.status == TaskStatus.PENDING
                    and t.kind == TaskKind.NORMAL
                    and not self._task_is_bulk_relay(t)
                    and int(t.demand_node) == int(s.node)
                    and self.is_task_serviceable_by_agent(str(aid), t)
                )
            ]
            if not onsite:
                continue

            # At a co-located group, preserve the normal task's deadline order.
            onsite.sort(key=lambda t: (int(t.deadline_step), str(t.task_id)))
            task = onsite[0]
            v2_service = v2_authorize_service_start(self, str(aid), task)
            if v2_service is not None and not bool(v2_service[0]):
                continue

            old_contract_truck = str(getattr(task, "route_contract_truck", "") or "")
            had_assist = str(aid) in self._planner_truck_assist_waypoint_by_truck
            contract_unreachable = False
            if old_contract_truck and old_contract_truck != str(aid):
                owner = self.state.agents.get(old_contract_truck)
                if owner is None or owner.kind != AgentKind.TRUCK or bool(owner.crashed):
                    contract_unreachable = True
                else:
                    owner_node = owner.node
                    if owner.transit is not None:
                        owner_node = int(owner.transit[1])
                    if owner_node is None:
                        contract_unreachable = True
                    else:
                        contract_unreachable = not bool(
                            np.isfinite(
                                self._decision_shortest_path_distance(
                                    int(owner_node), int(task.demand_node)
                                )
                            )
                        )

            local_goal = str(
                self._effective_goals.get(
                    str(aid), self._recommended_goals.get(str(aid), "")
                )
                or ""
            )
            hard_recovery_assigned = bool(
                self._truck_has_assigned_airborne_hard_recovery_request(
                    str(aid)
                )
            )
            # Minimal-impact activation: "closer" is not alone sufficient to
            # cancel a viable first-layer contract, because that can dismantle a
            # live emergency support chain.  Once layer 1 has explicitly rebound
            # the onsite task to this truck, however, execution must atomically
            # start unloading even when the ordinary post-action arrival trigger
            # would miss it.  This closes the seed110 N3/N4 service-start gap
            # without performing an early cross-contract takeover here.
            capture_required = bool(not hard_recovery_assigned)
            if not capture_required:
                continue
            if route_v2_active and old_contract_truck and old_contract_truck != str(aid):
                self.route_plan_v2_onsite_capture_contract_transfer_count += 1
            if route_v2_active and had_assist:
                self.route_plan_v2_onsite_capture_preempted_assist_count += 1

            # Transfer only this completed-in-place opportunity.  Pending remote
            # tasks keep their first-layer contracts and remain protected from churn.
            if route_v2_active:
                task.route_contract_truck = str(aid)
                task.route_contract_owner = str(aid)
                task.route_contract_uav_ids = ()
                task.route_contract_version = int(
                    max(getattr(task, "route_contract_version", 0), 0) + 1
                )
            task.assigned_to = str(aid)
            task.in_service_by = str(aid)
            task.status = TaskStatus.CLAIMED
            task.service_remaining = int(self._service_rounds(str(aid), task))
            self._effective_goals[str(aid)] = str(task.task_id)
            self._recommended_goals[str(aid)] = str(task.task_id)
            self._planner_truck_assist_waypoint_by_truck.pop(str(aid), None)
            self._planner_route_plan_stay_reason_by_agent[str(aid)] = (
                "atomic_onsite_routine_capture"
            )

            # Claimed tasks are hidden from the remaining agents.  Clearing stale
            # visible goals additionally prevents a low-level action from pursuing
            # the task during the same decision round.
            for other_aid in self.state.agents:
                if str(other_aid) == str(aid):
                    continue
                if str(self._effective_goals.get(str(other_aid), "")) == str(task.task_id):
                    self._effective_goals.pop(str(other_aid), None)
                if str(self._recommended_goals.get(str(other_aid), "")) == str(task.task_id):
                    self._recommended_goals.pop(str(other_aid), None)

            record_v2_service_start(self, str(aid), task)
            if route_v2_active:
                self.route_plan_v2_onsite_capture_count += 1
            started.append((str(aid), task))
            busy.add(str(aid))
        return started

    def _decay_task_lifeline_step(self, task: DeliveryTask) -> bool:
        if not bool(getattr(self.cfg, "task_lifeline_enabled", True)):
            return False
        if task.status in (TaskStatus.DELIVERED, TaskStatus.FAILED):
            return False
        base_decay = float(max(getattr(task, "lifeline_decay_rate", 0.0), 0.0))
        if base_decay <= 0.0:
            return False
        node_weather = self.hazards.node_weather(int(getattr(task, "demand_node", 0)))
        rain = float(getattr(node_weather, "rain", 0.0))
        wind = float(getattr(node_weather, "wind", 0.0))
        haz_factor = 1.0 + float(getattr(self.cfg, "task_lifeline_hazard_weight", 0.35)) * (rain / 40.0 + wind / 15.0)
        decay = float(max(base_decay * max(haz_factor, 0.2), 0.0))
        task.lifeline_current = float(max(float(getattr(task, "lifeline_current", task.lifeline_init)) - decay, 0.0))
        if float(task.lifeline_current) <= 1e-9:
            task.status = TaskStatus.FAILED
            task.failed_due_to_lifeline_zero = True
            task.failed_step = int(self.state.step_index)
            task.in_service_by = None
            task.service_remaining = 0
            return True
        return False

    def _advance_service_and_timeouts(
        self,
    ) -> Tuple[
        List[Tuple[str, DeliveryTask]],
        List[DeliveryTask],
        List[Tuple[str, DeliveryTask, float]],
    ]:
        delivered: List[Tuple[str, DeliveryTask]] = []
        timed_out: List[DeliveryTask] = []
        transfers: List[Tuple[str, DeliveryTask, float]] = []
        for task in self.state.tasks.values():
            if task.status in (TaskStatus.DELIVERED, TaskStatus.FAILED):
                continue

            if self._decay_task_lifeline_step(task):
                timed_out.append(task)
                continue

            if self.state.step_index > task.deadline_step:
                task.status = TaskStatus.FAILED
                task.failed_step = int(self.state.step_index)
                task.in_service_by = None
                task.service_remaining = 0
                timed_out.append(task)
                continue

            if (
                task.status == TaskStatus.CLAIMED
                and task.in_service_by is not None
                and int(task.service_remaining) > 0
            ):
                task.service_remaining = int(task.service_remaining) - 1
                if int(task.service_remaining) <= 0:
                    aid = str(task.in_service_by)
                    if getattr(task, "first_service_step", None) is None:
                        task.first_service_step = int(self.state.step_index)
                    service_agents = tuple(str(x) for x in getattr(task, "relay_service_agents", ()) if str(x)) if self._task_is_bulk_relay(task) else (aid,)
                    rem_before = float(
                        max(
                            float(
                                getattr(
                                    task,
                                    "remaining_demand_kg",
                                    self._task_demand_kg(task),
                                )
                            ),
                            0.0,
                        )
                    )
                    round_remaining_kg = float(rem_before)
                    transfers_this_round = []
                    for service_aid in service_agents:
                        if round_remaining_kg <= 1e-9:
                            break
                        # Cooperative UAVs unload sequentially against the same
                        # task balance so the final round cannot over-deliver.
                        task.remaining_demand_kg = float(round_remaining_kg)
                        transfer = float(self._consume_supply_for_service(service_aid, task))
                        transfers.append((service_aid, task, float(transfer)))
                        transfers_this_round.append((service_aid, transfer))
                        round_remaining_kg = float(max(round_remaining_kg - transfer, 0.0))
                    transfer = float(sum(value for _, value in transfers_this_round))

                    if transfer > 0.0:
                        rem_after = float(max(rem_before - float(transfer), 0.0))
                        task.remaining_demand_kg = float(rem_after)
                        task.fulfilled_mass_kg = float(max(float(getattr(task, "fulfilled_mass_kg", 0.0)) + float(transfer), 0.0))
                        task.demand_left = float(rem_after / max(float(getattr(self.cfg, "cargo_unit_kg", 200.0)), 1e-6))
                        if rem_after <= 1e-9:
                            v2_complete = v2_authorize_service_completion(self, aid, task, transfer=float(transfer))
                            if v2_complete is not None and not bool(v2_complete[0]):
                                task.status = TaskStatus.PENDING
                                task.assigned_to = None
                                task.in_service_by = None
                                task.service_remaining = 0
                                continue
                            task.status = TaskStatus.DELIVERED
                            task.delivered_by = aid
                            task.delivered_step = int(self.state.step_index)
                            task.failed_step = None
                            task.in_service_by = None
                            task.service_remaining = 0
                            task.relay_service_agents = ()
                            task.remaining_lifeline_at_service = float(getattr(task, "lifeline_current", 0.0))
                            record_v2_service_complete(self, str(aid), task)
                            delivered.append((aid, task))
                        else:
                            task.status = TaskStatus.PENDING
                            task.assigned_to = None
                            task.in_service_by = None
                            task.service_remaining = 0
                            task.relay_service_agents = ()
                            service_state = self.state.agents.get(str(aid), None)
                            if (
                                bool(
                                    getattr(
                                        self.cfg,
                                        "hrl_route_plan_routine_multiround_commitment_enabled",
                                        True,
                                    )
                                )
                                and service_state is not None
                                and service_state.kind == AgentKind.TRUCK
                                and task.kind == TaskKind.NORMAL
                            ):
                                previous = self._routine_multiround_service_commitment_by_truck.get(
                                    str(aid), None
                                )
                                self._routine_multiround_service_commitment_by_truck[str(aid)] = str(
                                    task.task_id
                                )
                                if str(previous) != str(task.task_id):
                                    self.routine_multiround_commitment_count += 1
                    else:
                        task.status = TaskStatus.PENDING
                        task.assigned_to = None
                        task.in_service_by = None
                        task.service_remaining = 0
                        task.relay_service_agents = ()
        return delivered, timed_out, transfers

    def _compute_task_semantic_metrics(self) -> Dict[str, float]:
        delivered_total = 0
        delivered_bulk = 0
        delivered_light = 0
        bulk_total = 0
        light_total = 0
        bulk_fulfilled_mass = 0.0
        bulk_demand_mass = 0.0
        failed_task_count = 0
        failed_due_to_lifeline_zero_count = 0
        service_delays_all: List[float] = []
        service_delays_bulk: List[float] = []
        service_delays_light: List[float] = []
        rem_lifeline_all: List[float] = []
        rem_lifeline_bulk: List[float] = []
        rem_lifeline_light: List[float] = []
        completion_time_all: List[float] = []
        completion_time_bulk: List[float] = []
        completion_time_light: List[float] = []
        time_critical_on_time_completed_count_total = 0
        weighted_service_num = 0.0
        weighted_service_den = 0.0

        for t in self.state.tasks.values():
            is_bulk = bool(self._task_is_routine_bulk(t))
            is_light = bool(self._task_is_time_critical_lightweight(t))
            urg = float(np.clip(float(getattr(t, "urgency_score", 0.5)), 0.0, 1.0))
            dem_kg = float(max(float(getattr(t, "demand_kg", self._task_demand_kg(t))), 1e-6))
            rem_kg = float(max(float(getattr(t, "remaining_demand_kg", max(dem_kg - float(getattr(t, "fulfilled_mass_kg", 0.0)), 0.0))), 0.0))
            fulfilled_ratio = float(np.clip((dem_kg - rem_kg) / dem_kg, 0.0, 1.0))

            if is_bulk:
                bulk_total += 1
                bulk_fulfilled_mass += float(dem_kg - rem_kg)
                bulk_demand_mass += float(dem_kg)
            if is_light:
                light_total += 1

            if t.status == TaskStatus.DELIVERED:
                delivered_total += 1
                if is_bulk:
                    delivered_bulk += 1
                if is_light:
                    delivered_light += 1
                dstep = int(getattr(t, "delivered_step", self.state.step_index))
                cstep = int(getattr(t, "created_step", 0))
                delay = float(max(dstep - cstep, 0) * self._dt_seconds)
                completion_time_all.append(delay)
                service_delays_all.append(delay)
                if is_bulk:
                    completion_time_bulk.append(delay)
                    service_delays_bulk.append(delay)
                if is_light:
                    completion_time_light.append(delay)
                    service_delays_light.append(delay)
                    if int(dstep) <= int(getattr(t, "deadline_step", dstep)):
                        time_critical_on_time_completed_count_total += 1
                rem_life = float(max(float(getattr(t, "remaining_lifeline_at_service", getattr(t, "lifeline_current", 0.0))), 0.0))
                rem_lifeline_all.append(rem_life)
                if is_bulk:
                    rem_lifeline_bulk.append(rem_life)
                if is_light:
                    rem_lifeline_light.append(rem_life)
            elif t.status == TaskStatus.FAILED:
                failed_task_count += 1

            if bool(getattr(t, "failed_due_to_lifeline_zero", False)):
                failed_due_to_lifeline_zero_count += 1

            if is_bulk:
                task_value = fulfilled_ratio
            else:
                if t.status == TaskStatus.DELIVERED:
                    life_init = float(max(float(getattr(t, "lifeline_init", 100.0)), 1e-6))
                    task_value = float(np.clip(float(getattr(t, "remaining_lifeline_at_service", 0.0)) / life_init, 0.0, 1.0))
                else:
                    task_value = 0.0
            weighted_service_num += float(urg * task_value)
            weighted_service_den += float(max(urg, 1e-6))

        routine_bulk_completion_rate = float(delivered_bulk / max(bulk_total, 1))
        time_critical_lightweight_completion_rate = float(delivered_light / max(light_total, 1))
        overall_completion_rate = float(delivered_total / max(len(self.state.tasks), 1))
        bulk_fulfilled_mass_ratio = float(np.clip(bulk_fulfilled_mass / max(bulk_demand_mass, 1e-6), 0.0, 1.0))
        weighted_service_score = float(weighted_service_num / max(weighted_service_den, 1e-6))
        time_critical_on_time_completion_rate = float(time_critical_on_time_completed_count_total / max(light_total, 1))

        return {
            "overall_completion_rate": float(overall_completion_rate),
            "routine_bulk_completion_rate": float(routine_bulk_completion_rate),
            "time_critical_lightweight_completion_rate": float(time_critical_lightweight_completion_rate),
            "time_critical_on_time_completion_rate": float(time_critical_on_time_completion_rate),
            "time_critical_on_time_completed_count_total": int(time_critical_on_time_completed_count_total),
            "time_critical_completion_time_mean_seconds": float(np.mean(completion_time_light)) if completion_time_light else 0.0,
            "routine_bulk_completion_time_mean_seconds": float(np.mean(completion_time_bulk)) if completion_time_bulk else 0.0,
            "overall_completion_time_mean_seconds": float(np.mean(completion_time_all)) if completion_time_all else 0.0,
            "bulk_fulfilled_mass_ratio": float(bulk_fulfilled_mass_ratio),
            "failed_task_count": int(failed_task_count),
            "failed_due_to_lifeline_zero_count": int(failed_due_to_lifeline_zero_count),
            "mean_remaining_lifeline_at_service": float(np.mean(rem_lifeline_all)) if rem_lifeline_all else 0.0,
            "mean_remaining_lifeline_bulk": float(np.mean(rem_lifeline_bulk)) if rem_lifeline_bulk else 0.0,
            "mean_remaining_lifeline_time_critical": float(np.mean(rem_lifeline_light)) if rem_lifeline_light else 0.0,
            "mean_remaining_lifeline_at_completion_time_critical": float(np.mean(rem_lifeline_light)) if rem_lifeline_light else 0.0,
            "average_service_delay": float(np.mean(service_delays_all)) if service_delays_all else 0.0,
            "average_service_delay_bulk": float(np.mean(service_delays_bulk)) if service_delays_bulk else 0.0,
            "average_service_delay_time_critical": float(np.mean(service_delays_light)) if service_delays_light else 0.0,
            "weighted_service_score": float(weighted_service_score),
            "routine_bulk_completed_count_total": int(delivered_bulk),
            "time_critical_lightweight_completed_count_total": int(delivered_light),
            "routine_bulk_failed_count_total": int(max(bulk_total - delivered_bulk, 0)),
            "time_critical_lightweight_failed_count_total": int(max(light_total - delivered_light, 0)),
        }

    def _agent_weather_sample(self, aid: str):
        """
        Unified weather sampling for communication and safety models.
        Priority is continuous position (`pos_xy`) during transit/flight.
        Fallback to nearest node only when continuous position is unavailable.
        """
        s = self.state.agents.get(str(aid))
        if s is None:
            return self.hazards.node_weather(0)
        if s.pos_xy is not None:
            return self.hazards.weather_at((float(s.pos_xy[0]), float(s.pos_xy[1])))
        return self.hazards.node_weather(int(s.node or 0))

    def _init_comm_blackout_protocol(self) -> None:
        """Build Scenario-C blackout zones from a dedicated exogenous RNG.

        The schedule is derived solely from the evaluation seed and initial task
        layout.  It therefore neither consumes the environment RNG nor changes
        the road/weather realization when communication is enabled.
        """
        self._comm_blackout_zones: List[Dict[str, Any]] = []
        self._comm_blackout_zone_task_ids: set[str] = set()
        self._comm_blackout_zone_node_ids: set[int] = set()
        self._comm_blackout_emergency_task_count: int = int(
            sum(1 for task in self.state.tasks.values() if task.kind == TaskKind.EMERGENCY)
        )
        self._comm_blackout_emergency_node_count: int = int(
            len(
                {
                    int(task.demand_node)
                    for task in self.state.tasks.values()
                    if task.kind == TaskKind.EMERGENCY
                }
            )
        )
        self._comm_blackout_zone_digest: str = self._comm_blackout_digest()
        self._comm_block_reason: Dict[str, str] = {}
        self._comm_blackout_rng = np.random.default_rng(int(self.cfg.seed) + 4_129_043)
        if (not bool(getattr(self.cfg, "enable_comm_blackout", False))) or str(self.cfg.scenario).upper() != "C":
            return
        if str(getattr(self.cfg, "comm_blackout_model", "regional_persistent_v1")).strip().lower() != "regional_persistent_v1":
            return

        emergency_by_node: Dict[int, List[Any]] = {}
        for task in self.state.tasks.values():
            if task.kind == TaskKind.EMERGENCY:
                emergency_by_node.setdefault(int(task.demand_node), []).append(task)
        emergency_nodes = sorted(emergency_by_node)
        if not emergency_nodes:
            return
        coverage = float(np.clip(getattr(self.cfg, "comm_blackout_emergency_coverage", 0.30), 0.0, 1.0))
        covered_count = int(min(len(emergency_nodes), max(1, int(np.ceil(coverage * len(emergency_nodes))))))
        zone_count = int(min(max(int(getattr(self.cfg, "comm_blackout_zone_count", 2)), 1), covered_count))

        # Pick spatially separated seed tasks, then grow compact task clusters
        # around them.  This makes the requested task-node coverage explicit
        # while yielding geographic blackout regions on M, L, and R maps.
        chosen: List[int] = []
        first = int(emergency_nodes[int(self._comm_blackout_rng.integers(0, len(emergency_nodes)))])
        chosen.append(first)
        while len(chosen) < zone_count:
            def nearest_sq(node_id: int) -> float:
                node = self.topology.nodes[int(node_id)]
                return min(
                    (float(node.x) - float(self.topology.nodes[int(seed)].x)) ** 2
                    + (float(node.y) - float(self.topology.nodes[int(seed)].y)) ** 2
                    for seed in chosen
                )
            candidates = [node_id for node_id in emergency_nodes if node_id not in chosen]
            chosen.append(max(candidates, key=nearest_sq))

        assignments: List[List[int]] = [[] for _ in chosen]
        remaining = list(emergency_nodes)
        for index, seed_node in enumerate(chosen):
            assignments[index].append(seed_node)
            remaining.remove(seed_node)
        while sum(len(items) for items in assignments) < covered_count and remaining:
            # Grow the currently smallest cluster with its nearest unassigned task.
            index = min(range(len(assignments)), key=lambda i: (len(assignments[i]), i))
            center_node = self.topology.nodes[int(chosen[index])]
            node_id = min(
                remaining,
                key=lambda candidate_node_id: (
                    (float(self.topology.nodes[int(candidate_node_id)].x) - float(center_node.x)) ** 2
                    + (float(self.topology.nodes[int(candidate_node_id)].y) - float(center_node.y)) ** 2,
                    int(candidate_node_id),
                ),
            )
            assignments[index].append(int(node_id))
            remaining.remove(node_id)

        xs = [float(node.x) for node in self.topology.nodes.values()]
        ys = [float(node.y) for node in self.topology.nodes.values()]
        diagonal = float(np.hypot(max(xs) - min(xs), max(ys) - min(ys))) if xs and ys else 1.0
        radius = float(max(diagonal * float(getattr(self.cfg, "comm_blackout_zone_radius_map_fraction", 0.12)), 1.0))
        duration = int(max(getattr(self.cfg, "comm_blackout_duration_steps", 6), 1))
        recovery = int(max(getattr(self.cfg, "comm_blackout_recovery_steps", 10), 1))
        cycle = int(duration + recovery)
        for index, (seed_node, members) in enumerate(zip(chosen, assignments)):
            node = self.topology.nodes[int(seed_node)]
            task_ids = {
                str(task.task_id)
                for member_node in members
                for task in emergency_by_node[int(member_node)]
            }
            self._comm_blackout_zones.append(
                {
                    "center_xy": (float(node.x), float(node.y)),
                    "radius_m": radius,
                    "phase_offset": int(index * cycle // max(zone_count, 1)),
                    "task_ids": task_ids,
                    "node_ids": set(int(member_node) for member_node in members),
                }
            )
            self._comm_blackout_zone_task_ids.update(task_ids)
            self._comm_blackout_zone_node_ids.update(int(member_node) for member_node in members)
        self._comm_blackout_zone_digest = self._comm_blackout_digest()

    def _comm_blackout_digest(self) -> str:
        """Return a behavior-independent fingerprint of the blackout world.

        ``comm_blackout_ratio`` is an outcome: it changes when algorithms visit
        different places.  This digest instead freezes zone geometry, schedule,
        and covered demand nodes so paired-method fairness can be audited before
        any route is executed.
        """

        zones = []
        for zone in getattr(self, "_comm_blackout_zones", []):
            center = tuple(float(value) for value in zone.get("center_xy", (0.0, 0.0)))
            zones.append(
                {
                    "center_xy": [round(center[0], 9), round(center[1], 9)],
                    "radius_m": round(float(zone.get("radius_m", 0.0)), 9),
                    "phase_offset": int(zone.get("phase_offset", 0)),
                    "task_ids": sorted(str(value) for value in zone.get("task_ids", set())),
                    "node_ids": sorted(int(value) for value in zone.get("node_ids", set())),
                }
            )
        payload = {
            "model": str(getattr(self.cfg, "comm_blackout_model", "")),
            "enabled": bool(getattr(self.cfg, "enable_comm_blackout", False)),
            "scenario": str(getattr(self.cfg, "scenario", "")).upper(),
            "start_step": int(getattr(self.cfg, "comm_blackout_start_step", 0)),
            "duration_steps": int(getattr(self.cfg, "comm_blackout_duration_steps", 0)),
            "recovery_steps": int(getattr(self.cfg, "comm_blackout_recovery_steps", 0)),
            "zones": zones,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _update_comm_blocked(self) -> None:
        # Hard-off switch for experiments without communication degradation.
        if not bool(getattr(self.cfg, "enable_comm_blackout", False)):
            self._comm_blackout_active_zone_count = 0
            for aid in self.state.agents:
                self.comm_blocked[aid] = False
                self._comm_block_reason[aid] = ""
            return

        model = str(getattr(self.cfg, "comm_blackout_model", "regional_persistent_v1")).strip().lower()
        if model == "regional_persistent_v1":
            start = int(max(getattr(self.cfg, "comm_blackout_start_step", 20), 0))
            duration = int(max(getattr(self.cfg, "comm_blackout_duration_steps", 6), 1))
            recovery = int(max(getattr(self.cfg, "comm_blackout_recovery_steps", 10), 1))
            cycle = int(duration + recovery)
            active_zones = []
            if int(self.state.step_index) >= start:
                elapsed = int(self.state.step_index) - start
                active_zones = [
                    zone for zone in self._comm_blackout_zones
                    if ((elapsed + int(zone["phase_offset"])) % cycle) < duration
                ]
            self._comm_blackout_active_zone_count = int(len(active_zones))
            active_task_ids = {
                str(task_id)
                for zone in active_zones
                for task_id in zone["task_ids"]
            }
            for aid, agent in self.state.agents.items():
                if agent.pos_xy is not None:
                    x, y = float(agent.pos_xy[0]), float(agent.pos_xy[1])
                else:
                    node = self.topology.nodes[int(agent.node or 0)]
                    x, y = float(node.x), float(node.y)
                physical_zone = any(
                    (x - float(zone["center_xy"][0])) ** 2 + (y - float(zone["center_xy"][1])) ** 2
                    <= float(zone["radius_m"]) ** 2
                    for zone in active_zones
                )
                # A command to an emergency task inside an active blackout
                # region also loses its end-to-end link.  This models damaged
                # local access/relay infrastructure before the vehicle itself
                # reaches the area, and prevents a geographically relevant C
                # condition from degenerating into a rarely visited disk.
                proposed_goal = getattr(self, "_recommended_goals", {}).get(str(aid), None)
                effective_goal = getattr(self, "_effective_goals", {}).get(str(aid), None)
                goal_zone = bool(
                    (proposed_goal is not None and str(proposed_goal) in active_task_ids)
                    or (effective_goal is not None and str(effective_goal) in active_task_ids)
                )
                self.comm_blocked[aid] = bool(physical_zone or goal_zone)
                self._comm_block_reason[aid] = (
                    "physical_and_goal_zone" if physical_zone and goal_zone
                    else "physical_zone" if physical_zone
                    else "goal_zone" if goal_zone
                    else ""
                )
            return

        # Legacy IID risk model retained solely for reproducing historical runs.
        if float(self.cfg.comm_block_prob) <= 0.0:
            self._comm_blackout_active_zone_count = 0
            for aid in self.state.agents:
                self.comm_blocked[aid] = False
                self._comm_block_reason[aid] = ""
            return

        for aid, s in self.state.agents.items():
            h = self._agent_weather_sample(aid)
            r_tilde = float(np.clip(h.rain / max(self.cfg.base_rainfall_mmh, 1e-6), 0.0, 1.0))
            e_tilde = float(np.clip(h.quake, 0.0, 1.0))
            p_blocked = float(np.clip(self.state.hazard.blocked_ratio, 0.0, 1.0))
            risk_score = float(0.45 * p_blocked + 0.35 * r_tilde + 0.20 * e_tilde)
            risk_weight = float(max(getattr(self.cfg, "comm_risk_score_weight", 0.55), 0.0))
            p = float(
                np.clip(
                    self.cfg.comm_block_prob + risk_weight * risk_score,
                    0.0,
                    0.95,
                )
            )
            self.comm_blocked[aid] = bool(self.rng.uniform() < p)
            self._comm_block_reason[aid] = "legacy_iid" if self.comm_blocked[aid] else ""
        self._comm_blackout_active_zone_count = 0

    def _uav_wind_failure_risk(self, aid: str) -> float:
        s = self.state.agents.get(str(aid))
        if s is None or s.kind != AgentKind.UAV or s.crashed:
            return 0.0
        # Use unified continuous-position weather sampling; only fallback to node
        # approximation when no continuous position exists.
        h = self._agent_weather_sample(str(aid))
        wind_mps = float(max(getattr(h, "wind", 0.0), 0.0))
        threshold = float(max(getattr(self.cfg, "wind_failure_threshold_mps", 16.0), 0.0))
        scale = float(np.clip(getattr(self.cfg, "wind_failure_risk_scale", 0.0), 0.0, 1.0))
        if scale <= 0.0 or wind_mps <= threshold:
            return 0.0
        exceed_ratio = float((wind_mps - threshold) / max(threshold, 1e-6))
        return float(np.clip(scale * exceed_ratio, 0.0, 1.0))

    def _uav_low_soc_failure_risk(self, aid: str) -> float:
        s = self.state.agents.get(str(aid))
        if s is None or s.kind != AgentKind.UAV or s.crashed:
            return 0.0
        if getattr(s, "follow_target", None) is not None:
            return 0.0
        batt = float(max(getattr(s, "battery", 0.0), 0.0))
        threshold = float(np.clip(getattr(self.cfg, "uav_low_soc_failure_threshold", 0.10), 0.0, 0.30))
        scale = float(np.clip(getattr(self.cfg, "uav_low_soc_failure_risk_scale", 0.0), 0.0, 1.0))
        if threshold <= 0.0 or scale <= 0.0 or batt >= threshold:
            return 0.0
        h = self._agent_weather_sample(str(aid))
        wind_mps = float(max(getattr(h, "wind", 0.0), 0.0))
        rain_mmh = float(max(getattr(h, "rain", 0.0), 0.0))
        m_load = float(max(getattr(s, "payload_kg_current", 0.0), 0.0))
        soc_severity = float(np.clip((threshold - batt) / max(threshold, 1e-6), 0.0, 1.0))
        env_amp = float(1.0 + 0.04 * wind_mps + 0.02 * rain_mmh + 0.012 * m_load)
        return float(np.clip(scale * soc_severity * env_amp, 0.0, 1.0))

    def _node_xy(self, node_id: int) -> Tuple[float, float]:
        node = self.topology.nodes[int(node_id)]
        return float(node.x), float(node.y)

    def _nearest_node(self, x: float, y: float) -> int:
        best_node = 0
        best_dist = 1e18
        for node_id, n in self.topology.nodes.items():
            d = (n.x - x) ** 2 + (n.y - y) ** 2
            if d < best_dist:
                best_dist = d
                best_node = int(node_id)
        return best_node

    def _has_open_emergency_tasks(self) -> bool:
        for t in self.state.tasks.values():
            if not self._task_is_uav_delivery(t):
                continue
            if t.status in (TaskStatus.PENDING, TaskStatus.CLAIMED):
                return True
        return False

    def _advance_truck_transit(self, aid: str) -> None:
        s = self.state.agents[aid]
        if s.transit is None:
            return
        src, dst, remain = s.transit
        src = int(src)
        dst = int(dst)
        remain_prev = float(max(float(remain), 0.0))
        remain = max(0.0, remain_prev - float(self._dt_seconds))

        # Continuous truck position along edge to avoid teleport in rendering.
        dist = float(self.topology.edge_distance(src, dst))
        payload_kg = float(self._truck_transport_mass_kg(str(aid)))
        speed = float(max(self._truck_speed_mps(src, dst, payload_kg=payload_kg), 1e-6))
        full_time = float(max(dist / speed, 1e-6))
        moved_time = float(max(remain_prev - remain, 0.0))
        moved_dist = float(np.clip(speed * moved_time, 0.0, max(dist, 0.0)))
        s.lifetime_distance_m = float(getattr(s, "lifetime_distance_m", 0.0) + moved_dist)

        if remain <= 0.0:
            s.node = dst
            s.pos_xy = self._node_xy(dst)
            s.transit = None
            self._truck_last_arrived_from[str(aid)] = int(src)
        else:
            p0 = self._node_xy(src)
            p1 = self._node_xy(dst)
            progress = float(np.clip(1.0 - remain / full_time, 0.0, 1.0))
            x = (1.0 - progress) * p0[0] + progress * p1[0]
            y = (1.0 - progress) * p0[1] + progress * p1[1]
            s.node = src
            s.pos_xy = (float(x), float(y))
            s.transit = (src, dst, float(remain))

    def _start_truck_move(self, aid: str, target_node: int) -> bool:
        s = self.state.agents[aid]
        if s.transit is not None:
            return False
        if s.node is None:
            return False
        src = int(s.node)
        dst = int(target_node)
        if int(dst) not in self.topology.adjacency.get(int(src), set()):
            return False

        # Shared-cognition gate: if edge already known blocked in shared map,
        # planner should not intentionally command crossing.
        if self._decision_is_blocked(int(src), int(dst)):
            return False

        # Physical world check: unknown blocked edges are discovered when truck
        # attempts to enter and fails. This creates cognition lag without UAV warning.
        if bool(self.topology.is_blocked(int(src), int(dst))):
            self._unknown_blocked_edge_hit_step += 1
            self._unknown_blocked_edge_hit_total += 1
            self._update_shared_map_edge_observation((int(src), int(dst)), True, source="truck_contact")
            return False
        if not v2_edge_accessible(self, int(src), int(dst)):
            self._unknown_blocked_edge_hit_step += 1
            self._unknown_blocked_edge_hit_total += 1
            self._update_shared_map_edge_observation((int(src), int(dst)), True, source="physical_v2")
            return False

        dist = self.topology.edge_distance(src, dst)
        payload_kg = float(self._truck_transport_mass_kg(str(aid)))
        travel_time = dist / max(self._truck_speed_mps(src, dst, payload_kg=payload_kg), 1e-6)
        s.transit = (src, dst, float(travel_time))
        return True

    def _record_invalid_action(
        self,
        aid: str,
        raw_action: object,
        normalized_action: object,
        *,
        validation_layer: str,
        reason_code: str,
        reason_detail: str,
        local_repair_attempted: bool = False,
        local_repair_succeeded: bool = False,
        fallback_action: object | None = None,
        source_code_location: str = "",
    ) -> None:
        rec = make_invalid_action_record(
            self,
            str(aid),
            raw_action,
            normalized_action,
            validation_layer=validation_layer,
            reason_code=reason_code,
            reason_detail=reason_detail,
            local_repair_attempted=local_repair_attempted,
            local_repair_succeeded=local_repair_succeeded,
            fallback_action=fallback_action,
            source_code_location=source_code_location,
        )
        self.invalid_action_records.append(rec)

    def pre_dispatch_validate_actions(self, actions: Dict[str, object]) -> Dict[str, object]:
        """Validate selected low-level actions against the current env state.

        Invalid selected actions are repaired to the local safe no-op instead of
        being submitted to ``step`` and counted as environment invalid actions.
        """
        out: Dict[str, object] = dict(actions or {})
        for aid, st in self.state.agents.items():
            raw = out.get(str(aid), None)
            result = validate_action_for_dispatch(self, str(aid), raw)
            if result.valid:
                if result.normalized_action is not None:
                    out[str(aid)] = result.normalized_action
                continue

            fallback = result.fallback_action
            if fallback is None:
                fallback = safe_noop_for_agent_state(st)
            repair_result = validate_action_for_dispatch(self, str(aid), fallback)
            if not repair_result.valid:
                fallback = safe_noop_for_agent_state(st)
            self.pre_dispatch_rejected_count_total += 1
            self.safe_noop_fallback_count_total += 1
            if validate_action_for_dispatch(self, str(aid), fallback).valid:
                self.pre_dispatch_repair_success_count_total += 1
            self._record_invalid_action(
                str(aid),
                raw,
                result.normalized_action,
                validation_layer="pre_dispatch",
                reason_code=str(result.reason_code or "UNKNOWN_INVALID_REASON"),
                reason_detail=str(result.reason_detail),
                local_repair_attempted=True,
                local_repair_succeeded=True,
                fallback_action=fallback,
                source_code_location=str(result.source_code_location),
            )
            out[str(aid)] = fallback
        return out

    def _truck_speed_mps(self, src: int, dst: int, payload_kg: float = 0.0) -> float:
        e = self.topology.edge_attr(src, dst)
        slope_norm = float(
            np.clip(
                0.5
                * (self.topology.nodes[src].slope_norm + self.topology.nodes[dst].slope_norm),
                0.0,
                1.0,
            )
        )
        slope_deg = 35.0 * slope_norm
        rough = float(np.clip(e.roughness_norm, 0.0, 1.0))
        payload = float(max(payload_kg, 0.0))
        den = (1.0 + 0.015 * slope_deg) * (1.0 + 0.55 * rough) * (1.0 + 0.00035 * payload)
        v2_mult = float(v2_truck_speed_multiplier(self, int(src), int(dst)))
        if not np.isfinite(v2_mult):
            return 0.0
        den *= float(max(v2_mult, 1e-6))
        return float(max(0.5, float(self.cfg.truck_speed_mps) / max(den, 1e-6)))

    def _erc_v2_authorized_truck_support(self, aid: str, mode: str) -> bool:
        if not bool(getattr(self, "_erc_v2_command_gate_enabled", False)):
            return True
        batch = getattr(self, "_erc_v2_command_batch", None)
        if CommandValidator.is_truck_support_authorized(batch, str(aid), str(mode)):
            return True
        self.unauthorized_support_attempt_count += 1
        self.unauthorized_support_blocked_count += 1
        if str(mode) == "recovery":
            self.unauthorized_recovery_attempt_count += 1
            self.unauthorized_recovery_blocked_count += 1
        return False

    def _erc_v2_truck_command_target(self, aid: str, kind: str = "") -> Optional[int]:
        if not bool(getattr(self, "_erc_v2_command_gate_enabled", False)):
            return None
        batch = getattr(self, "_erc_v2_command_batch", None)
        cmd = getattr(batch, "truck_commands", {}).get(str(aid)) if batch is not None else None
        if cmd is None:
            return None
        if kind and str(getattr(cmd, "kind", "")) != str(kind):
            return None
        target = getattr(cmd, "target_node", None)
        if target is None:
            target = getattr(cmd, "support_point", None)
        return None if target is None else int(target)

    def _erc_v2_authorized_uav_launch(self, aid: str, task_id: Optional[str]) -> bool:
        if not bool(getattr(self, "_erc_v2_command_gate_enabled", False)):
            return True
        batch = getattr(self, "_erc_v2_command_batch", None)
        if CommandValidator.is_uav_launch_authorized(batch, str(aid), None if task_id is None else str(task_id)):
            return True
        self.command_rejected_count += 1
        self.command_rejected_reason_launch_unauthorized_count += 1
        return False

    def _apply_uav_action(
        self, aid: str, act: UAVAction
    ) -> Tuple[bool, float, bool, float, float, bool, bool]:
        """
        Returns:
            (invalid_action, moved_dist_m, new_bind, headwind_mps, rain_mmh, queue_waiting, sortie_limited)
        """
        s = self.state.agents[aid]
        if s.crashed:
            return False, 0.0, False, 0.0, 0.0, False, False
        if s.pos_xy is None:
            if s.node is not None:
                s.pos_xy = self._node_xy(int(s.node))
            else:
                s.pos_xy = (0.0, 0.0)
        force_hover = False
        sortie_limited = False
        vx = float(act.vx)
        vy = float(act.vy)

        step_now = int(self.state.step_index)
        commit_steps = int(max(getattr(self.cfg, "uav_bind_commit_steps", 4), 1))
        commit_tid = self._uav_bind_commit_target.get(str(aid), None)
        commit_until = int(self._uav_bind_commit_until_step.get(str(aid), -1))

        # Follow-mode release (hard safety gate).
        if s.follow_target is not None and bool(act.takeoff):
            if self._uav_launch_block_cooldown_active(str(aid)):
                return False, 0.0, False, 0.0, 0.0, False, False
            dwell_rem = int(max(self._uav_post_bind_dwell_remaining.get(str(aid), 0), 0))
            if dwell_rem > 0:
                launch_goal_id = self._effective_goals.get(str(aid), self._recommended_goals.get(str(aid), None))
                launch_task = self.state.tasks.get(str(launch_goal_id)) if launch_goal_id is not None else None
                self._note_uav_task_reject(str(aid), launch_task, "post_bind_dwell")
                self._note_unsafe_launch_attempt(str(aid), reason="post_bind_dwell")
                self._uav_mark_forced_recovery(str(aid))
                return False, 0.0, False, 0.0, 0.0, False, False
            launch_goal_id = self._effective_goals.get(str(aid), self._recommended_goals.get(str(aid), None))
            launch_task = self.state.tasks.get(str(launch_goal_id)) if launch_goal_id is not None else None
            if not self._erc_v2_authorized_uav_launch(str(aid), str(launch_goal_id) if launch_goal_id is not None else None):
                self._note_unsafe_launch_attempt(str(aid), reason="v2_launch_without_command")
                return False, 0.0, False, 0.0, 0.0, False, False
            transfer_target_tid = self._uav_transfer_target_for_task(str(aid), launch_task)
            if transfer_target_tid is not None:
                # A transfer hint is a fallback, not permission to move away
                # from an already launchable delivery.  Prefer the complete
                # physical delivery gate at the UAV's current carrier; only
                # relocate when that delivery is not yet safe/feasible.
                delivery_ok, delivery_reason, _delivery_force_recovery = (
                    self._uav_launch_gate_check(
                        str(aid),
                        task=launch_task,
                        count_reject=False,
                    )
                )
                delivery_preferred = bool(
                    _UAV_DIRECT_DELIVERY_OVER_TRANSFER_ENABLED
                    and delivery_ok
                    and (
                        str(delivery_reason).startswith("direct_safe")
                        or str(delivery_reason).startswith("rendezvous_safe")
                    )
                )
                if delivery_preferred and launch_task is not None:
                    task_xy = self._node_xy(int(launch_task.demand_node))
                    current_xy = self._agent_xy(str(aid))
                    long_leg_distance = float(
                        np.hypot(
                            float(current_xy[0]) - float(task_xy[0]),
                            float(current_xy[1]) - float(task_xy[1]),
                        )
                    )
                    delivery_preferred = bool(
                        long_leg_distance
                        >= _UAV_DIRECT_DELIVERY_LONG_LEG_THRESHOLD_M
                    )
                transfer_advances_task = True
                if (
                    _UAV_NONPROGRESS_TRANSFER_FALLBACK_ENABLED
                    and launch_task is not None
                ):
                    target_truck = self.state.agents.get(
                        str(transfer_target_tid)
                    )
                    if target_truck is not None:
                        task_xy = self._node_xy(int(launch_task.demand_node))
                        current_xy = self._agent_xy(str(aid))
                        target_xy = self._agent_xy(str(transfer_target_tid))
                        current_task_dist = float(
                            np.hypot(
                                float(current_xy[0]) - float(task_xy[0]),
                                float(current_xy[1]) - float(task_xy[1]),
                            )
                        )
                        target_task_dist = float(
                            np.hypot(
                                float(target_xy[0]) - float(task_xy[0]),
                                float(target_xy[1]) - float(task_xy[1]),
                            )
                        )
                        progress_tolerance = float(
                            max(
                                getattr(
                                    self.cfg,
                                    "uav_delivery_radius_m",
                                    100.0,
                                ),
                                100.0,
                            )
                        )
                        transfer_advances_task = bool(
                            target_task_dist
                            <= current_task_dist + progress_tolerance
                        )
                if not transfer_advances_task:
                    reachable_routine_remains = any(
                        self._truck_has_reachable_serviceable_normal(
                            str(truck_id)
                        )
                        for truck_id, truck_state in self.state.agents.items()
                        if truck_state.kind == AgentKind.TRUCK
                    )
                    lifeline_init = float(
                        max(
                            getattr(launch_task, "lifeline_init", 100.0),
                            1e-6,
                        )
                    )
                    lifeline_ratio = float(
                        max(
                            getattr(
                                launch_task,
                                "lifeline_current",
                                lifeline_init,
                            ),
                            0.0,
                        )
                        / lifeline_init
                    )
                    critical_ratio = float(
                        np.clip(
                            getattr(
                                self.cfg,
                                "hrl_timecritical_lifeline_critical_ratio",
                                0.35,
                            ),
                            0.0,
                            1.0,
                        )
                    )
                    fallback_released = bool(
                        (not reachable_routine_remains)
                        or lifeline_ratio <= critical_ratio
                    )
                    delivery_preferred = bool(
                        delivery_ok
                        and fallback_released
                        and (
                            str(delivery_reason).startswith("direct_safe")
                            or str(delivery_reason).startswith(
                                "rendezvous_safe"
                            )
                        )
                    )
                    if not delivery_preferred:
                        # Release only this stalled UAV-side contract so the
                        # planner can reauction the pending emergency task to
                        # another loaded UAV.  The current carrier remains
                        # docked and its truck route is not redirected.
                        if str(
                            getattr(launch_task, "route_contract_owner", "")
                        ) == str(aid):
                            launch_task.route_contract_owner = None
                            launch_task.route_contract_truck = None
                            launch_task.route_contract_uav_ids = ()
                            launch_task.route_contract_version = int(
                                max(
                                    getattr(
                                        launch_task,
                                        "route_contract_version",
                                        0,
                                    ),
                                    0,
                                )
                                + 1
                            )
                        launch_task.assigned_to = None
                        self._uav_sortie_contract_task.pop(str(aid), None)
                        self._uav_sortie_contract_version.pop(str(aid), None)
                        self._uav_transfer_target_truck.pop(str(aid), None)
                        self._uav_transfer_target_task.pop(str(aid), None)
                        self._uav_last_launch_reason[str(aid)] = ""
                        self._effective_goals[str(aid)] = str(
                            getattr(s, "follow_target", "") or ""
                        ) or None
                        return (
                            False,
                            0.0,
                            False,
                            0.0,
                            0.0,
                            False,
                            False,
                        )
                if delivery_preferred:
                    launch_truck_id = str(
                        getattr(s, "follow_target", "") or ""
                    )
                    if self._commit_uav_delivery_launch(
                        str(aid),
                        launch_task,
                        str(delivery_reason),
                        launch_truck_id=launch_truck_id,
                    ):
                        self._uav_transfer_target_truck.pop(str(aid), None)
                        self._uav_transfer_target_task.pop(str(aid), None)
                        transfer_target_tid = None
                # A transfer hop is permitted when it is explicitly tied to
                # the current task.  This is a meaningful sortie: it moves
                # the UAV to the truck that can continue that task.
                if transfer_target_tid is not None:
                    transfer_ok, transfer_reason = self._uav_transfer_takeoff_gate_check(
                        str(aid),
                        str(transfer_target_tid),
                        task=launch_task,
                    )
                    if transfer_ok:
                        s.follow_target = None
                        s.replenish_timer = 0
                        s.uav_reload_timer = 0
                        self._uav_bind_commit_target[str(aid)] = str(transfer_target_tid)
                        commit_tid = str(transfer_target_tid)
                        commit_steps_transfer = int(max(getattr(self.cfg, "hrl_uav_task_transfer_commit_steps", commit_steps), 1))
                        self._uav_bind_commit_until_step[str(aid)] = int(step_now + commit_steps_transfer - 1)
                        commit_until = int(self._uav_bind_commit_until_step[str(aid)])
                        self._uav_last_launch_reason[str(aid)] = str(transfer_reason)
                        self._uav_sortie_relaxed_latch[str(aid)] = False
                        self._record_uav_launch(str(aid))
                        self.uav_safe_launch_count_total += 1
                        self.uav_transfer_launch_count_total += 1
                    else:
                        self._note_unsafe_launch_attempt(str(aid), reason=str(transfer_reason))
                        return False, 0.0, False, 0.0, 0.0, False, False
            else:
                if bool(getattr(self.cfg, "uav_post_bind_force_recharge", True)):
                    dep_thr = float(self._uav_force_takeoff_battery_threshold(task=launch_task))
                    if float(getattr(s, "battery", 0.0)) + 1e-9 < dep_thr:
                        self._note_uav_task_reject(str(aid), launch_task, "post_bind_recharge")
                        self._note_unsafe_launch_attempt(str(aid), reason="post_bind_recharge")
                        self._uav_mark_forced_recovery(str(aid))
                        return False, 0.0, False, 0.0, 0.0, False, False
                if bool(getattr(self.cfg, "uav_post_bind_force_reload", True)) and (not self._uav_loaded(str(aid))):
                    self._note_uav_task_reject(str(aid), launch_task, "post_bind_reload")
                    self._note_unsafe_launch_attempt(str(aid), reason="post_bind_reload")
                    self._uav_mark_forced_recovery(str(aid))
                    return False, 0.0, False, 0.0, 0.0, False, False

                launch_ok, launch_reason, force_recovery = self._uav_launch_gate_check(
                    str(aid),
                    task=launch_task,
                    count_reject=True,
                )
                if not launch_ok:
                    self._note_unsafe_launch_attempt(str(aid), reason=str(launch_reason))
                    if force_recovery:
                        rec_reason = "low_battery" if str(launch_reason) == "below_launch_min" else "launch_gate"
                        self._uav_mark_forced_recovery(str(aid), reason=rec_reason)
                    return False, 0.0, False, 0.0, 0.0, False, False

                launch_truck_id = str(getattr(s, "follow_target", "")) if getattr(s, "follow_target", None) is not None else ""
                delivery_launch_committed = self._commit_uav_delivery_launch(
                    str(aid),
                    launch_task,
                    str(launch_reason),
                    launch_truck_id=launch_truck_id,
                )
                if delivery_launch_committed:
                    self._uav_transfer_target_task.pop(str(aid), None)
                    # Keep the post-transfer continuation until the task is
                    # completed/invalidated.  A launch accepted here can still
                    # be rebound in the same step by the physical recovery
                    # gate; clearing early would lose the original delivery.
                # A rendezvous-safe launch means the sortie needs a moving
                # recovery plan after service; it should not immediately take
                # over the UAV and pull it away from the emergency task.
                if bool(force_recovery) and not str(launch_reason).startswith("rendezvous_safe"):
                    self._uav_mark_forced_recovery(str(aid), reason="launch_rendezvous")

        # Follow binding with commit lock.
        if s.follow_target is None and act.bind_truck_id is not None:
            requested_tid = str(act.bind_truck_id)
            tid = requested_tid
            if commit_tid is not None and int(step_now) <= int(commit_until) and str(requested_tid) != str(commit_tid):
                tid = str(commit_tid)
            ts = self.state.agents.get(tid)
            if ts is None or ts.kind != AgentKind.TRUCK or bool(getattr(ts, "crashed", False)):
                if self._uav_recovery_required(str(aid)):
                    self.uav_rendezvous_fail_count_total += 1
                self._last_uav_invalid_reason[str(aid)] = "INVALID_RECOVERY_ANCHOR"
                return True, 0.0, False, 0.0, 0.0, False, False
            if not self._truck_has_follow_slot(str(tid), exclude_aid=str(aid)):
                # Queue full: reject bind, force hover, keep normal air-side energy consumption.
                force_hover = True
            if not self._truck_can_accept_uav_payload(str(tid), str(aid)):
                # A loaded transfer/recovery may not push the receiving truck
                # above its 3000 kg material capacity.
                force_hover = True
                self._last_uav_invalid_reason[str(aid)] = "TRUCK_PAYLOAD_CAPACITY_FULL"
            txy = ts.pos_xy if ts.pos_xy is not None else self._node_xy(int(ts.node or 0))
            uxy = s.pos_xy if s.pos_xy is not None else self._node_xy(int(s.node or 0))
            d = float(np.hypot(float(uxy[0]) - float(txy[0]), float(uxy[1]) - float(txy[1])))
            bind_window = float(self._uav_bind_window_m(ts))
            recovery_mode_before_bind = bool(self._uav_recovery_required(str(aid)))
            if (not force_hover) and d <= bind_window:
                transfer_task_id = str(
                    self._uav_transfer_target_task.get(str(aid), "")
                    or self._uav_sortie_contract_task.get(str(aid), "")
                ).strip()
                was_task_transfer = bool(
                    transfer_task_id
                    and str(
                        self._uav_last_launch_reason.get(str(aid), "")
                    )
                    == "truck_transfer"
                )
                s.follow_target = tid
                self._uav_last_launch_reason[str(aid)] = ""
                self._uav_sortie_relaxed_latch[str(aid)] = False
                s.sortie_distance_m = 0.0
                self._uav_post_bind_dwell_remaining[str(aid)] = int(max(getattr(self.cfg, "uav_post_bind_min_dwell_steps", 0), 0))
                self._uav_bind_commit_target[str(aid)] = None
                self._uav_bind_commit_until_step[str(aid)] = -1
                needs_reload = bool(
                    int(getattr(s, "carried_emergency_units", 0)) < int(max(self.cfg.uav_max_emergency_units, 1))
                    or bool(getattr(s, "uav_needs_reload_flag", False))
                )
                if needs_reload or float(s.battery) < 1.0:
                    s.replenish_timer = int(max(0, int(self.cfg.replenish_freeze_steps)))
                if recovery_mode_before_bind:
                    self.uav_rendezvous_success_count_total += 1
                    record_v2_recovery(self, str(aid), "RECOVERED", "recovery_bind_window")
                self._uav_transfer_target_truck.pop(str(aid), None)
                self._uav_transfer_target_task.pop(str(aid), None)
                if was_task_transfer:
                    if _UAV_TASK_TRANSFER_BIND_RELEASE_ENABLED:
                        self._uav_sortie_contract_task.pop(str(aid), None)
                        self._uav_sortie_contract_version.pop(str(aid), None)
                        self._uav_post_transfer_contract_task.pop(
                            str(aid), None
                        )
                    elif _UAV_POST_TRANSFER_CONTRACT_CONTINUATION_ENABLED:
                        self._uav_post_transfer_contract_task[str(aid)] = str(
                            transfer_task_id
                        )
                return False, 0.0, True, 0.0, 0.0, False, False
            if force_hover:
                if self._uav_recovery_required(str(aid)):
                    self.uav_rendezvous_fail_count_total += 1
                vx = 0.0
                vy = 0.0
            else:
                recovery_mode = bool(self._uav_recovery_required(str(aid)))
                if recovery_mode:
                    self.uav_rendezvous_fail_count_total += 1
                self._uav_bind_commit_target[str(aid)] = str(tid)
                self._uav_bind_commit_until_step[str(aid)] = int(step_now + commit_steps - 1)
                if recovery_mode:
                    # In recovery mode, out-of-window bind request is converted to chase
                    # so UAV keeps closing with truck instead of paying repeated invalid penalties.
                    return False, 0.0, False, 0.0, 0.0, False, False
                self._last_uav_invalid_reason[str(aid)] = "RENDEZVOUS_NOT_FEASIBLE"
                return True, 0.0, False, 0.0, 0.0, False, False

        # If landed/following truck: physics overridden, no free movement.
        if s.follow_target is not None:
            return False, 0.0, False, 0.0, 0.0, False, False

        vx = float(vx)
        vy = float(vy)

        force_recover_thr = float(np.clip(getattr(self.cfg, "uav_low_battery_force_recover_threshold", 0.25), 0.0, 1.0))
        low_lock_thr = float(np.clip(getattr(self.cfg, "uav_low_battery_goal_lock_threshold", 0.35), 0.0, 1.0))
        if float(getattr(s, "battery", 0.0)) <= force_recover_thr:
            self._uav_mark_forced_recovery(str(aid), reason="low_battery")
        if s.follow_target is None and bool(self._uav_needs_reload(str(aid))):
            self._uav_mark_forced_recovery(str(aid), reason="needs_reload")

        # In bind-commit window or forced recovery mode, UAV must keep approaching truck rendezvous.
        commit_tid = self._uav_bind_commit_target.get(str(aid), None)
        commit_until = int(self._uav_bind_commit_until_step.get(str(aid), -1))
        commit_active = bool(commit_tid is not None and int(step_now) <= int(commit_until))
        forced_rth_latched = bool(self._uav_forced_rth_latch.get(str(aid), False))
        forced_recovery = bool(
            float(getattr(s, "battery", 0.0)) <= force_recover_thr
            or (
                forced_rth_latched
                and (
                    not self._uav_sortie_delivery_recovery_bypass_active(
                        str(aid)
                    )
                )
            )
        )
        if commit_active or forced_recovery:
            target_tid = str(commit_tid) if commit_active else None
            if target_tid is not None and not self._truck_can_accept_uav_payload(str(target_tid), str(aid)):
                target_tid = None
            if target_tid is None or self.state.agents.get(str(target_tid), None) is None:
                require_stock = bool(self._uav_needs_reload(str(aid)))
                target_tid, _ = self._nearest_truck_from_xy(
                    s.pos_xy if s.pos_xy is not None else self._node_xy(int(s.node or 0)),
                    require_emergency_stock=require_stock,
                    require_follow_slot=True,
                    exclude_aid=str(aid),
                )
                if (
                    target_tid is None
                    and require_stock
                    and bool(getattr(self.cfg, "uav_reload_at_depot_enabled", True))
                ):
                    depot_xy = self._node_xy(0)
                    uxy = s.pos_xy if s.pos_xy is not None else self._node_xy(int(s.node or 0))
                    dx = float(depot_xy[0] - float(uxy[0]))
                    dy = float(depot_xy[1] - float(uxy[1]))
                    norm = float(np.hypot(dx, dy))
                    if norm > 1e-6:
                        vmax = float(max(getattr(self.cfg, "uav_max_speed_mps", 1.0), 1e-6))
                        vx = float(dx / norm * vmax)
                        vy = float(dy / norm * vmax)
            if target_tid is not None:
                txy = self._agent_xy(str(target_tid))
                uxy = s.pos_xy if s.pos_xy is not None else self._node_xy(int(s.node or 0))
                dx = float(txy[0] - float(uxy[0]))
                dy = float(txy[1] - float(uxy[1]))
                norm = float(np.hypot(dx, dy))
                bind_window = float(self._uav_bind_window_m(self.state.agents.get(str(target_tid), None)))
                if (
                    norm <= bind_window
                    and self._truck_has_follow_slot(str(target_tid), exclude_aid=str(aid))
                    and self._truck_can_accept_uav_payload(str(target_tid), str(aid))
                ):
                    transfer_task_id = str(
                        self._uav_transfer_target_task.get(str(aid), "")
                        or self._uav_sortie_contract_task.get(str(aid), "")
                    ).strip()
                    was_task_transfer = bool(
                        transfer_task_id
                        and str(
                            self._uav_last_launch_reason.get(str(aid), "")
                        )
                        == "truck_transfer"
                    )
                    recovery_mode_before_bind = bool(self._uav_recovery_required(str(aid)))
                    s.follow_target = str(target_tid)
                    s.sortie_distance_m = 0.0
                    self._uav_recovery_requested_truck.pop(str(aid), None)
                    self._uav_post_bind_dwell_remaining[str(aid)] = int(max(getattr(self.cfg, "uav_post_bind_min_dwell_steps", 0), 0))
                    self._uav_bind_commit_target[str(aid)] = None
                    self._uav_bind_commit_until_step[str(aid)] = -1
                    if str(self._uav_last_launch_reason.get(str(aid), "")) == "truck_transfer":
                        self.uav_transfer_bind_count_total += 1
                    self._uav_transfer_target_truck.pop(str(aid), None)
                    self._uav_transfer_target_task.pop(str(aid), None)
                    if was_task_transfer:
                        if _UAV_TASK_TRANSFER_BIND_RELEASE_ENABLED:
                            self._uav_sortie_contract_task.pop(str(aid), None)
                            self._uav_sortie_contract_version.pop(
                                str(aid), None
                            )
                            self._uav_post_transfer_contract_task.pop(
                                str(aid), None
                            )
                        elif _UAV_POST_TRANSFER_CONTRACT_CONTINUATION_ENABLED:
                            self._uav_post_transfer_contract_task[
                                str(aid)
                            ] = str(transfer_task_id)
                    needs_reload = bool(
                        int(getattr(s, "carried_emergency_units", 0)) < int(max(self.cfg.uav_max_emergency_units, 1))
                        or bool(getattr(s, "uav_needs_reload_flag", False))
                    )
                    if needs_reload or float(s.battery) < 1.0:
                        s.replenish_timer = int(max(0, int(self.cfg.replenish_freeze_steps)))
                    if recovery_mode_before_bind:
                        self.uav_rendezvous_success_count_total += 1
                        record_v2_recovery(self, str(aid), "RECOVERED", "forced_recovery_bind")
                    return False, 0.0, True, 0.0, 0.0, False, False
                if norm > 1e-6:
                    vmax = float(max(getattr(self.cfg, "uav_max_speed_mps", 1.0), 1e-6))
                    vx = float(dx / norm * vmax)
                    vy = float(dy / norm * vmax)
                    hold_thr_base = float(np.clip(getattr(self.cfg, "uav_recovery_idle_hold_threshold", 0.18), 0.0, 1.0))
                    hold_min_dist = float(max(getattr(self.cfg, "uav_recovery_idle_hold_min_dist_m", 600.0), 0.0))
                    truck_speed = float(max(getattr(self.cfg, "truck_speed_mps", 0.0), 1e-6))
                    bind_radius = float(max(getattr(self.cfg, "uav_bind_radius_m", 170.0), 0.0))
                    idle_drain = float(max(getattr(self.cfg, "uav_idle_discharge_per_step", 0.0), 0.0))
                    steps_to_bind = float(max(norm - bind_radius, 0.0) / max(truck_speed * max(self._dt_seconds, 1e-6), 1e-6))
                    hold_thr_dynamic = float(min(0.95, steps_to_bind * idle_drain + 0.03))
                    hold_thr = float(max(hold_thr_base, hold_thr_dynamic))
                    if (
                        forced_recovery
                        and bool(getattr(self.cfg, "truck_support_uav_recovery_enabled", True))
                        and float(getattr(s, "battery", 0.0)) <= hold_thr
                        and norm >= hold_min_dist
                    ):
                        vx = 0.0
                        vy = 0.0

        # Optional rule takeover: auto-approach when target enters monitor radius.
        tgt = self._pbrs_target_task(aid)
        auto_approach_enabled = bool(getattr(self.cfg, "uav_auto_approach_enabled", False)) or bool(
            getattr(self.cfg, "enable_auto_approach", False)
        )
        if (
            auto_approach_enabled
            and (not force_hover)
            and tgt is not None
            and s.pos_xy is not None
        ):
            tn = self.topology.nodes[int(tgt.demand_node)]
            dx = float(tn.x - s.pos_xy[0])
            dy = float(tn.y - s.pos_xy[1])
            d = float(np.hypot(dx, dy))
            if d <= float(self.cfg.uav_monitor_radius_m) and d > 1e-6:
                ux = dx / d
                uy = dy / d
                vx = float(ux * self.cfg.uav_max_speed_mps)
                vy = float(uy * self.cfg.uav_max_speed_mps)
        cmd_speed = float(np.hypot(vx, vy))
        headwind = 0.0
        rain = 0.0
        if cmd_speed > 1e-6 and s.pos_xy is not None:
            wx, wy = self.hazards.wind_vector_at(s.pos_xy)
            dir_x = vx / cmd_speed
            dir_y = vy / cmd_speed
            headwind = float(max(0.0, -(wx * dir_x + wy * dir_y)))
            weather = self.hazards.weather_at(s.pos_xy)
            rain = float(max(0.0, weather.rain))

        # 3.2 UAV speed attenuation formula with dynamic cargo coupling.
        # Normalize by the configured heavy-lift rating so a full 150 kg load
        # has the same dimensionless load ratio as any other rated platform.
        m_load_kg = float(max(getattr(s, "payload_kg_current", 0.0), 0.0))
        payload_capacity_kg = float(max(getattr(self.cfg, "uav_payload_capacity_kg", 150.0), 1e-6))
        payload_ratio = float(np.clip(m_load_kg / payload_capacity_kg, 0.0, 1.0))
        f_load = float(max(0.25, 1.0 - 0.16 * payload_ratio))
        f_rain = float(max(0.35, 1.0 - 0.018 * rain))
        f_head = float(max(0.30, 1.0 - 0.03 * headwind))
        v_uav = float(self.cfg.uav_max_speed_mps) * f_load * f_rain * f_head

        if cmd_speed > v_uav and cmd_speed > 1e-6:
            scale = v_uav / max(cmd_speed, 1e-6)
            vx *= scale
            vy *= scale

        dt = float(self._dt_seconds)
        terminal_delivery_commit = False
        # A launch accepted under the authoritative cooperative sortie
        # contract must not be turned away immediately outside the delivery
        # capture band by the legacy direct-return predictor.  Override only
        # the final physical step, only for the same loaded pending task, and
        # only when that step leaves the configured emergency reserve intact.
        if bool(
            getattr(
                self.cfg,
                "uav_terminal_delivery_commitment_enabled",
                True,
            )
        ):
            contract_tid = self._uav_sortie_contract_task.get(str(aid), None)
            contract_task = (
                self.state.tasks.get(str(contract_tid), None)
                if contract_tid is not None
                else None
            )
            if (
                contract_task is not None
                and contract_task.status == TaskStatus.PENDING
                and self._task_is_uav_delivery(contract_task)
                and bool(self._uav_loaded_for_task(str(aid), contract_task))
                and s.pos_xy is not None
            ):
                task_xy = self._node_xy(int(contract_task.demand_node))
                dx_task = float(task_xy[0] - float(s.pos_xy[0]))
                dy_task = float(task_xy[1] - float(s.pos_xy[1]))
                d_task = float(np.hypot(dx_task, dy_task))
                base_capture = float(
                    max(getattr(self.cfg, "uav_delivery_radius_m", 0.0), 0.0)
                )
                motion_capture = float(
                    max(
                        getattr(
                            self.cfg,
                            "uav_delivery_capture_motion_factor",
                            0.80,
                        ),
                        0.0,
                    )
                    * max(float(getattr(self.cfg, "uav_max_speed_mps", 0.0)), 0.0)
                    * max(dt, 0.0)
                )
                capture_radius = float(max(base_capture, motion_capture))
                terminal_envelope = float(max(v_uav * dt + capture_radius, 0.0))
                if (
                    d_task > capture_radius + 1e-9
                    and d_task <= terminal_envelope + 1e-9
                    and d_task > 1e-9
                ):
                    terminal_step = float(min(v_uav * dt, d_task))
                    terminal_origin = (float(s.pos_xy[0]), float(s.pos_xy[1]))
                    terminal_dir = (float(dx_task / d_task), float(dy_task / d_task))
                    terminal_destination = (
                        terminal_origin[0] + terminal_dir[0] * terminal_step,
                        terminal_origin[1] + terminal_dir[1] * terminal_step,
                    )
                    required = float(
                        self._uav_energy_cost_fraction(
                            str(aid),
                            terminal_step,
                            terminal_origin,
                            destination=terminal_destination,
                        )
                    )
                    reserve = float(
                        np.clip(
                            getattr(
                                self.cfg,
                                "uav_terminal_delivery_min_reserve_fraction",
                                0.20,
                            ),
                            0.0,
                            1.0,
                        )
                    )
                    service_buffer = float(
                        max(getattr(self.cfg, "service_battery_buffer", 0.0), 0.0)
                    )
                    if float(getattr(s, "battery", 0.0)) + 1e-9 >= float(
                        required + reserve + service_buffer
                    ):
                        vx = float(dx_task / d_task * v_uav)
                        vy = float(dy_task / d_task * v_uav)
                        terminal_delivery_commit = True
        world_lim = float(max(getattr(self.cfg, "map_size_m", 3000.0), 1.0))
        ox = float(s.pos_xy[0])
        oy = float(s.pos_xy[1])
        nx = float(np.clip(ox + vx * dt, 0.0, world_lim))
        ny = float(np.clip(oy + vy * dt, 0.0, world_lim))
        dist = float(np.hypot(nx - ox, ny - oy))
        delivery_commitment_active = bool(
                forced_rth_latched
                and not forced_recovery
                and self._uav_sortie_delivery_recovery_bypass_active(str(aid))
        )
        if delivery_commitment_active and dist > 1e-9:
            contract_tid = self._uav_sortie_contract_task.get(str(aid), None)
            contract_task = (
                self.state.tasks.get(str(contract_tid), None)
                if contract_tid is not None
                else None
            )
            if contract_task is None:
                delivery_commitment_active = False
            else:
                task_xy = self._node_xy(int(contract_task.demand_node))
                task_dx = float(task_xy[0] - ox)
                task_dy = float(task_xy[1] - oy)
                # Keep the outbound-only bypass tied to motion that closes the
                # accepted task leg; arbitrary movement still uses recovery
                # prediction even while a stale latch is present.
                delivery_commitment_active = bool(
                    (nx - ox) * task_dx + (ny - oy) * task_dy >= -1e-9
                )

        # Predictive safety gate while airborne:
        # veto this movement if projected post-step battery cannot preserve
        # conservative rendezvous margin to nearest truck.
        if dist > 1e-9:
            batt_now = float(max(getattr(s, "battery", 0.0), 0.0))
            est_step_cost = float(
                self._uav_actual_movement_energy(
                    str(aid),
                    (ox, oy),
                    (nx, ny),
                )
            )
            recovery_tid_new, d_back_new = self._nearest_truck_from_xy((nx, ny))
            if np.isfinite(d_back_new):
                recovery_buf = float(max(getattr(self.cfg, "uav_recovery_distance_buffer_m", 600.0), 0.0))
                reserve = float(np.clip(getattr(self.cfg, "uav_emergency_reserve_fraction", 0.20), 0.0, 1.0))
                rendez_margin = float(np.clip(getattr(self.cfg, "uav_rendezvous_margin_fraction", 0.10), 0.0, 1.0))
                req_back_new = float(
                    self._uav_energy_cost_fraction(
                        str(aid),
                        float(d_back_new + recovery_buf),
                        (nx, ny),
                        destination=self._uav_extended_destination(
                            (nx, ny),
                            self._agent_xy(str(recovery_tid_new)) if recovery_tid_new is not None else (nx, ny),
                            float(d_back_new + recovery_buf),
                        ),
                        payload_override=self._uav_expected_payload_after_task(
                            str(aid),
                            self.state.tasks.get(str(self._uav_sortie_contract_task.get(str(aid), ""))),
                        ),
                    )
                )
                projected_post = float(batt_now - est_step_cost)
                required_post = float(req_back_new + reserve + rendez_margin)
                if projected_post + 1e-9 < required_post:
                    if terminal_delivery_commit or delivery_commitment_active:
                        # An accepted sortie may bypass this full-return
                        # predictor while its outbound delivery leg remains
                        # feasible; terminal approach keeps its explicit
                        # reserve counter as before.
                        if terminal_delivery_commit:
                            self.uav_terminal_delivery_commitment_count = int(
                                getattr(
                                    self,
                                    "uav_terminal_delivery_commitment_count",
                                    0,
                                )
                            ) + 1
                    else:
                        self._uav_mark_forced_recovery(str(aid), reason="low_battery")
                        nx, ny = ox, oy
                        vx, vy = 0.0, 0.0
                        dist = 0.0

        # Low-battery hard lock: do not allow moves that increase truck distance.
        if (float(getattr(s, "battery", 0.0)) < low_lock_thr) and (not forced_recovery) and (not commit_active):
            _, d0 = self._nearest_truck_from_xy((ox, oy))
            _, d1 = self._nearest_truck_from_xy((nx, ny))
            if np.isfinite(d0) and np.isfinite(d1) and (d1 > d0 + 1e-6):
                nx, ny = ox, oy
                vx, vy = 0.0, 0.0
                dist = 0.0

        # Forced-recovery motion hard gate:
        # do not execute a movement step that would likely push battery below a
        # minimal safety floor before rendezvous can occur.
        if forced_recovery and dist > 1e-9:
            est_step_cost = float(
                self._uav_actual_movement_energy(
                    str(aid),
                    (ox, oy),
                    (nx, ny),
                )
            )
            batt_now = float(max(getattr(s, "battery", 0.0), 0.0))
            force_thr = float(np.clip(getattr(self.cfg, "uav_low_battery_force_recover_threshold", 0.25), 0.0, 1.0))
            safety_floor = float(max(0.05, 0.25 * force_thr))
            if batt_now <= float(est_step_cost + safety_floor):
                nx, ny = ox, oy
                vx, vy = 0.0, 0.0
                dist = 0.0

        # Optional legacy hard sortie envelope. The main paper configuration
        # uses battery/energy/recovery feasibility instead of a fixed distance
        # ceiling, because large real-city maps make a static sortie cap
        # dominate the intended cooperative recovery behavior.
        if self._legacy_sortie_cap_enabled():
            max_sortie = float(max(getattr(self.cfg, "uav_max_sortie_m", 0.0), 0.0))
            remaining_sortie = float(max(max_sortie - float(s.sortie_distance_m), 0.0))
            if dist > 1e-9:
                if remaining_sortie <= 1e-9:
                    nx, ny = ox, oy
                    vx, vy = 0.0, 0.0
                    dist = 0.0
                    sortie_limited = True
                elif dist > remaining_sortie:
                    ratio = float(remaining_sortie / dist)
                    nx = float(ox + (nx - ox) * ratio)
                    ny = float(oy + (ny - oy) * ratio)
                    vx = float(vx * ratio)
                    vy = float(vy * ratio)
                    dist = float(remaining_sortie)
                    sortie_limited = True

        s.vel_xy = (vx, vy)
        s.pos_xy = (nx, ny)
        s.node = self._nearest_node(nx, ny)
        s.sortie_distance_m += float(dist)
        s.lifetime_distance_m += float(dist)
        if (forced_recovery or commit_active) and dist > 1e-9:
            energy_before_motion = float(getattr(s, "battery", 0.0))
            energy_after_estimate = float(
                energy_before_motion
                - float(self._uav_actual_movement_energy(str(aid), (ox, oy), (nx, ny)))
            )
            motion_weather = self.hazards.weather_at((float(ox), float(oy)))
            record_v2_recovery_motion(
                self,
                str(aid),
                target_truck_id=str(target_tid) if "target_tid" in locals() and target_tid is not None else None,
                target_anchor=int(self.state.agents[str(target_tid)].node)
                if "target_tid" in locals()
                and target_tid is not None
                and self.state.agents.get(str(target_tid), None) is not None
                and getattr(self.state.agents[str(target_tid)], "node", None) is not None
                else None,
                old_xy=(float(ox), float(oy)),
                new_xy=(float(nx), float(ny)),
                distance_m=float(dist),
                reason="forced_recovery" if forced_recovery else "bind_commit",
                wind_speed=float(getattr(motion_weather, "wind", 0.0)),
                wind_direction=0.0,
                rain=float(getattr(motion_weather, "rain", 0.0)),
                visibility=float(max(10.0 - float(getattr(motion_weather, "rain", 0.0)) / 10.0, 0.5)),
                energy_before=energy_before_motion,
                energy_after=energy_after_estimate,
                reason_codes=("RECOVERY_PENDING" if forced_recovery else "BIND_COMMIT",),
            )

        docked_depot = self._uav_try_dock_depot(str(aid))
        if docked_depot:
            return False, dist, True, float(headwind), float(rain), bool(force_hover), bool(sortie_limited)
        return False, dist, False, float(headwind), float(rain), bool(force_hover), bool(sortie_limited)

    def _sync_follow_and_charge(self, aid: str) -> float:
        s = self.state.agents[aid]
        if s.follow_target is None:
            return 0.0

        dock_is_depot = bool(str(s.follow_target) == DEPOT_DOCK_ID)
        ts = None if dock_is_depot else self.state.agents.get(s.follow_target)
        if (not dock_is_depot) and (ts is None or ts.kind != AgentKind.TRUCK):
            s.follow_target = None
            s.replenish_timer = 0
            s.uav_reload_timer = 0
            return 0.0

        if dock_is_depot:
            s.node = 0
            s.pos_xy = self._node_xy(0)
        elif ts is not None and ts.transit is not None:
            # Keep following while truck is moving.
            src, dst, remain = ts.transit
            a = self.topology.nodes[int(src)]
            b = self.topology.nodes[int(dst)]
            full = self.topology.edge_distance(int(src), int(dst))
            truck_payload_kg = float(self._truck_transport_mass_kg(str(getattr(ts, "agent_id", ""))))
            speed = float(
                max(
                    self._truck_speed_mps(int(src), int(dst), payload_kg=truck_payload_kg),
                    1e-6,
                )
            )
            progress = 1.0 - float(remain / max(full / speed, 1e-6))
            progress = float(np.clip(progress, 0.0, 1.0))
            x = (1.0 - progress) * a.x + progress * b.x
            y = (1.0 - progress) * a.y + progress * b.y
            s.pos_xy = (float(x), float(y))
            s.node = int(dst) if progress >= 0.5 else int(src)
        else:
            s.node = int(ts.node or 0)
            s.pos_xy = self._node_xy(int(s.node))
        s.vel_xy = (0.0, 0.0)
        self._uav_terminal_battery_rescue_active.discard(str(aid))

        # Battery recharge can happen while docked/following.
        need_recharge = bool(float(s.battery) < 1.0)
        freeze_steps = int(max(0, int(getattr(self.cfg, "replenish_freeze_steps", 0))))
        charge_gain = 0.0
        if need_recharge:
            if freeze_steps <= 0:
                before = float(s.battery)
                step_gain = float(max(getattr(self.cfg, "uav_charge_rate_per_step", 0.0), 0.0))
                s.battery = float(min(1.0, float(s.battery) + step_gain))
                charge_gain = float(max(0.0, s.battery - before))
                s.replenish_timer = 0
            else:
                if int(s.replenish_timer) <= 0:
                    s.replenish_timer = int(freeze_steps)
                if int(s.replenish_timer) > 0:
                    s.replenish_timer = int(s.replenish_timer) - 1
                    if int(s.replenish_timer) == 0:
                        before = float(s.battery)
                        s.battery = 1.0
                        charge_gain = float(max(0.0, s.battery - before))
        else:
            s.replenish_timer = 0

        if charge_gain > 1e-9:
            self.uav_recharge_count_total += 1

        # UAV payload reload is a separate service stage.  Legacy tasks keep
        # the single emergency package.  A v2 BULK_RELAY stop instead loads a
        # capacity-bounded bulk chunk while preserving the task as NORMAL.
        goal_id_reload = self._effective_goals.get(
            str(aid), self._recommended_goals.get(str(aid), None)
        )
        goal_task_reload = (
            self.state.tasks.get(str(goal_id_reload))
            if goal_id_reload is not None
            else None
        )
        desired_supply_type = (
            "normal"
            if self._task_is_bulk_relay(goal_task_reload)
            else "emergency"
        )
        current_supply_type = str(
            getattr(s, "payload_supply_type", "emergency")
        ).strip().lower()
        if (
            current_supply_type != desired_supply_type
            and float(getattr(s, "payload_kg_current", 0.0)) > 1e-9
        ):
            returned_kg = float(max(getattr(s, "payload_kg_current", 0.0), 0.0))
            if ts is not None:
                self._sync_truck_inventory_fields(ts)
                if current_supply_type == "normal":
                    ts.bulk_inventory_kg_current = float(
                        max(getattr(ts, "bulk_inventory_kg_current", 0.0), 0.0)
                        + returned_kg
                    )
                else:
                    ts.timecritical_inventory_kg_current = float(
                        max(
                            getattr(
                                ts, "timecritical_inventory_kg_current", 0.0
                            ),
                            0.0,
                        )
                        + returned_kg
                    )
                self._sync_truck_inventory_fields(ts)
            s.carried_emergency_units = 0
            s.payload_kg_current = 0.0
            s.cargo = 0.0
            s.uav_needs_reload_flag = True

        need_reload = bool(
            bool(getattr(s, "uav_needs_reload_flag", False))
            or int(getattr(s, "carried_emergency_units", 0)) < int(max(self.cfg.uav_max_emergency_units, 1))
            or float(getattr(s, "payload_kg_current", 0.0)) < float(self.cfg.emergency_task_demand_kg) - 1e-9
            or current_supply_type != desired_supply_type
        )
        if need_reload:
            # A partial package left after service is not discarded. Return it
            # to the docked truck before loading the next sortie.
            residual_kg = float(max(getattr(s, "payload_kg_current", 0.0), 0.0))
            if residual_kg > 1e-9 and bool(getattr(s, "uav_needs_reload_flag", False)):
                if ts is not None:
                    self._sync_truck_inventory_fields(ts)
                    if current_supply_type == "normal":
                        ts.bulk_inventory_kg_current = float(
                            max(getattr(ts, "bulk_inventory_kg_current", 0.0), 0.0)
                            + residual_kg
                        )
                    else:
                        ts.timecritical_inventory_kg_current = float(
                            max(getattr(ts, "timecritical_inventory_kg_current", 0.0), 0.0)
                            + residual_kg
                        )
                    self._sync_truck_inventory_fields(ts)
                s.carried_emergency_units = 0
                s.payload_kg_current = 0.0
                s.cargo = 0.0
            if int(s.uav_reload_timer) <= 0:
                s.uav_reload_timer = int(max(1, int(getattr(self.cfg, "uav_reload_service_steps", 1))))
            if int(s.uav_reload_timer) > 0:
                s.uav_reload_timer = int(s.uav_reload_timer) - 1
                if int(s.uav_reload_timer) == 0:
                    req_units = int(max(1, int(getattr(self.cfg, "uav_max_emergency_units", 1))))
                    reload_success = False
                    loaded_kg = 0.0
                    if dock_is_depot and bool(getattr(self.cfg, "uav_reload_at_depot_enabled", True)):
                        # Depot has unlimited stock in the current paper setting.
                        loaded_kg = float(
                            self._uav_relay_payload_capacity_kg(goal_task_reload)
                            if desired_supply_type == "normal"
                            else self._timecritical_supply_unit_kg()
                        )
                        reload_success = True
                    elif ts is not None:
                        self._sync_truck_inventory_fields(ts)
                        if desired_supply_type == "normal":
                            capacity_kg = float(self._uav_relay_payload_capacity_kg(goal_task_reload))
                            bulk_kg = float(
                                max(
                                    getattr(
                                        ts, "bulk_inventory_kg_current", 0.0
                                    ),
                                    0.0,
                                )
                            )
                            loaded_kg = float(min(capacity_kg, bulk_kg))
                            if loaded_kg > 1e-9:
                                ts.bulk_inventory_kg_current = float(
                                    max(bulk_kg - loaded_kg, 0.0)
                                )
                                self._sync_truck_inventory_fields(ts)
                                reload_success = True
                        else:
                            req_kg = float(max(float(req_units) * self._timecritical_supply_unit_kg(), 1e-6))
                            if float(getattr(ts, "timecritical_inventory_kg_current", 0.0)) >= req_kg - 1e-9:
                                ts.timecritical_inventory_kg_current = float(max(float(getattr(ts, "timecritical_inventory_kg_current", 0.0)) - req_kg, 0.0))
                                self._sync_truck_inventory_fields(ts)
                                loaded_kg = float(self._timecritical_supply_unit_kg())
                                reload_success = True

                    if reload_success:
                        s.carried_emergency_units = int(req_units)
                        s.payload_kg_current = float(
                            loaded_kg
                            if loaded_kg > 1e-9
                            else self._timecritical_supply_unit_kg()
                        )
                        s.payload_supply_type = str(desired_supply_type)
                        s.uav_needs_reload_flag = False
                        self._sync_uav_payload_fields(s)
                        self.uav_reload_count_total += 1
                    else:
                        s.uav_needs_reload_flag = True
                        self._sync_uav_payload_fields(s)
                        self.uav_reload_wait_steps_total += 1
                else:
                    self.uav_reload_wait_steps_total += 1

        if int(self._uav_post_bind_dwell_remaining.get(str(aid), 0)) > 0:
            self._uav_post_bind_dwell_remaining[str(aid)] = int(self._uav_post_bind_dwell_remaining[str(aid)] - 1)

        return float(charge_gain)

    def _update_uav_energy_and_crash(
        self, aid: str, moved_dist_m: float, headwind_mps: float, rain_mmh: float
    ) -> bool:
        s = self.state.agents[aid]
        if s.kind != AgentKind.UAV or s.crashed:
            return False
        if s.follow_target is not None:
            # Charging already handled in follow sync.
            return False
        energy_before = float(getattr(s, "battery", 0.0))
        if moved_dist_m > 1e-6:
            new_xy = s.pos_xy if s.pos_xy is not None else self._node_xy(int(s.node or 0))
            old_xy = getattr(self, "_uav_motion_old_xy", {}).get(str(aid), new_xy)
            s.battery -= float(
                self._uav_energy_cost_fraction(
                    str(aid),
                    float(moved_dist_m),
                    (float(old_xy[0]), float(old_xy[1])),
                    destination=(float(new_xy[0]), float(new_xy[1])),
                )
            )
            record_v2_energy_deduction(
                self,
                str(aid),
                energy_before=energy_before,
                energy_after=float(getattr(s, "battery", 0.0)),
                distance_m=float(moved_dist_m),
                reason="uav_runtime_movement",
            )
        else:
            idle_scale = 1.0
            if bool(self._uav_forced_rth_latch.get(str(aid), False)):
                idle_scale = float(np.clip(getattr(self.cfg, "uav_recovery_idle_discharge_scale", 0.25), 0.0, 1.0))
            s.battery -= float(self.cfg.uav_idle_discharge_per_step) * idle_scale
            record_v2_energy_deduction(
                self,
                str(aid),
                energy_before=energy_before,
                energy_after=float(getattr(s, "battery", 0.0)),
                distance_m=0.0,
                reason="uav_runtime_idle",
            )
        terminal_floor = float(np.clip(getattr(self.cfg, "uav_terminal_failure_battery_floor", 0.0), 0.0, 0.20))
        if s.battery <= terminal_floor:
            # Hard survival guard for paper safety runs:
            # prefer forced recovery / rendezvous over terminal crash when truck support exists.
            support_enabled = bool(getattr(self.cfg, "truck_support_uav_recovery_enabled", True))
            forced_recovery_active = bool(self._uav_forced_rth_latch.get(str(aid), False))
            if support_enabled and forced_recovery_active:
                bind_latency_steps = int(max(getattr(self.cfg, "uav_forced_recovery_bind_latency_steps", 0), 0))
                forced_step = int(self._uav_forced_rth_start_step.get(str(aid), -1))
                latency_satisfied = bool(
                    bind_latency_steps <= 0
                    or (forced_step >= 0 and int(self.state.step_index) - forced_step >= bind_latency_steps)
                )
                uxy = s.pos_xy if s.pos_xy is not None else self._node_xy(int(s.node or 0))
                if latency_satisfied:
                    tid, d_back = self._nearest_truck_from_xy(
                        uxy,
                        require_follow_slot=True,
                        exclude_aid=str(aid),
                    )
                    if tid is not None:
                        ts = self.state.agents.get(str(tid), None)
                        bind_window = float(self._uav_bind_window_m(ts))
                        if np.isfinite(d_back) and float(d_back) <= float(1.5 * bind_window):
                            s.battery = float(max(terminal_floor + 1e-3, 1e-3))
                            s.follow_target = str(tid)
                            s.sortie_distance_m = 0.0
                            self._uav_terminal_battery_rescue_active.discard(str(aid))
                            self._uav_recovery_requested_truck.pop(str(aid), None)
                            record_v2_recovery(self, str(aid), "RECOVERED", "terminal_guard_bind")
                            self._uav_post_bind_dwell_remaining[str(aid)] = int(max(getattr(self.cfg, "uav_post_bind_min_dwell_steps", 0), 0))
                            self._uav_bind_commit_target[str(aid)] = None
                            self._uav_bind_commit_until_step[str(aid)] = -1
                            return False

            hard_guard_enabled = bool(getattr(self.cfg, "uav_hard_recovery_battery_guard", True))
            if support_enabled and (forced_recovery_active or hard_guard_enabled):
                s.battery = float(max(terminal_floor + 1e-3, 1e-3))
                self._uav_mark_forced_recovery(str(aid), reason="low_battery")
                record_v2_recovery(self, str(aid), "RECOVERY_PENDING", "low_battery_guard")
                if str(aid) not in self._uav_terminal_battery_rescue_active:
                    self._uav_terminal_battery_rescue_active.add(str(aid))
                    self.uav_terminal_battery_rescue_count_total += 1
                s.vel_xy = (0.0, 0.0)
                s.sortie_distance_m = 0.0
                return False

            s.battery = float(max(terminal_floor, 0.0))
            s.crashed = True
            s.follow_target = None
            s.sortie_distance_m = 0.0
            record_v2_recovery(self, str(aid), "UAV_DROP", "terminal_battery")
            return True
        return False

    def step(self, action: JointAction) -> StepResult:
        # Step-1: transition skeleton only (no full physics yet).
        if self.state.done:
            return StepResult(
                state=self.state,
                rewards={aid: 0.0 for aid in self.state.agents},
                terminated=True,
                truncated=False,
                info={"reason": "already_done"},
            )

        rewards = {

            aid: float(self.cfg.reward_step_penalty) for aid in self.state.agents
        }
        reward_step_total = float(self.cfg.reward_step_penalty) * float(
            len(self.state.agents)
        )
        reward_invalid_total = 0.0
        reward_idle_total = 0.0
        reward_delivery_total = 0.0
        reward_timeout_total = 0.0
        reward_pbrs_total = 0.0
        reward_crash_total = 0.0
        reward_discover_total = 0.0
        reward_docking_total = 0.0
        reward_pickup_total = 0.0
        reward_delivery_shared_total = 0.0
        reward_uav_emergency_bonus_total = 0.0
        self.state.step_index += 1

        invalid_action_count = 0
        invalid_action_count_uav = 0
        invalid_action_count_truck = 0
        moved_dists: Dict[str, float] = {aid: 0.0 for aid in self.state.agents}
        moved_headwind: Dict[str, float] = {aid: 0.0 for aid in self.state.agents}
        moved_rain: Dict[str, float] = {aid: 0.0 for aid in self.state.agents}
        self._uav_motion_old_xy = {
            aid: (
                tuple(float(v) for v in s.pos_xy)
                if s.pos_xy is not None
                else self._node_xy(int(s.node or 0))
            )
            for aid, s in self.state.agents.items()
            if s.kind == AgentKind.UAV
        }
        pre_step_battery: Dict[str, float] = {
            aid: float(s.battery)
            for aid, s in self.state.agents.items()
            if s.kind == AgentKind.UAV
        }
        step_follow_bind_count = 0
        step_follow_steps = 0
        step_follow_charge_energy = 0.0
        step_low_battery_events = 0
        step_low_battery_return_success = 0
        step_forced_takeoff_full = 0
        step_forced_rth_count = 0
        step_queue_wait_steps = 0
        step_uav_energy_used = 0.0
        step_crash_count = 0
        step_battery_depletion_count = 0
        step_pbrs_switch_count = 0
        step_uav_discovered_blocked = 0
        step_uav_goal_task_count = 0
        step_uav_terminal_zone_count = 0
        step_uav_delivery_zone_count = 0
        step_sortie_limit_hits = 0
        step_wind_failure_event_count = 0
        step_wind_failure_risk_accum = 0.0
        step_truck_replenish_events = 0
        step_truck_empty_trip_count = 0
        step_uav_empty_flight_count = 0
        step_truck_forward_support_distance = 0.0

        self._shared_awareness_step_reset()

        reload_count_before = int(self.uav_reload_count_total)
        replenish_count_before = int(self.truck_replenish_count_total)
        normal_block_before = int(self.normal_tasks_blocked_by_supply_count)
        emergency_block_before = int(self.emergency_tasks_blocked_by_supply_count)
        recharge_count_before = int(self.uav_recharge_count_total)
        safe_launch_before = int(self.uav_safe_launch_count_total)
        launch_count_before = int(self.uav_launch_count_total)
        unsafe_launch_attempt_before = int(self.uav_unsafe_launch_attempt_count_total)
        unsafe_launch_before = int(self.uav_unsafe_launch_block_count_total)
        low_batt_illegal_launch_before = int(self.uav_low_battery_illegal_launch_count_total)
        forced_recovery_before = int(self.uav_forced_recovery_count_total)
        forced_recovery_low_batt_before = int(self.uav_forced_recovery_due_to_low_battery_count_total)
        rendezvous_success_before = int(self.uav_rendezvous_success_count_total)
        rendezvous_fail_before = int(self.uav_rendezvous_fail_count_total)
        truck_recovery_support_before = int(self.truck_recovery_support_count_total)
        truck_forward_support_before = int(self.truck_forward_support_count_total)
        truck_forward_support_dist_before = float(self.truck_forward_support_distance_total)
        island_completed_before = int(self.island_task_completed_count_total)
        island_delivery_before = int(self.uav_island_delivery_count_total)
        island_recovery_before = int(self.uav_island_recovery_success_count_total)

        uav_reject_below_before = int(self.uav_task_reject_below_launch_min_count)
        uav_reject_not_loaded_before = int(self.uav_task_reject_not_loaded_count)
        uav_reject_margin_before = int(self.uav_task_reject_recovery_margin_count)
        uav_reject_horizon_before = int(self.uav_task_reject_horizon_count)
        uav_reject_comm_before = int(self.uav_task_reject_comm_block_count)
        uav_reject_corridor_before = int(self.uav_task_reject_corridor_count)
        uav_launch_direct_before = int(self.uav_launch_direct_safe_count)
        uav_launch_rendezvous_before = int(self.uav_launch_rendezvous_safe_count)
        uav_launch_rendezvous_relaxed_before = int(self.uav_launch_rendezvous_safe_relaxed_count)
        uav_launch_block_unsafe_before = int(self.uav_launch_block_unsafe_count)
        uav_launch_gate_enter_before = int(self.uav_launch_gate_enter_count)
        uav_launch_gate_direct_before = int(self.uav_launch_gate_direct_safe_count)
        uav_launch_gate_rendezvous_before = int(self.uav_launch_gate_rendezvous_safe_count)
        uav_launch_gate_rendezvous_relaxed_before = int(self.uav_launch_gate_rendezvous_safe_relaxed_count)
        uav_launch_gate_block_below_before = int(self.uav_launch_gate_block_below_launch_min_count)
        uav_launch_gate_block_margin_before = int(self.uav_launch_gate_block_recovery_margin_count)
        uav_launch_gate_block_corridor_before = int(self.uav_launch_gate_block_corridor_count)
        uav_launch_gate_block_other_before = int(self.uav_launch_gate_block_other_count)
        truck_emergency_block_guard_before = int(self.truck_emergency_blocked_by_normal_guard_count)
        truck_emergency_relief_override_before = int(self.truck_emergency_relief_override_count)
        truck_emergency_serviceable_before = int(self.truck_emergency_serviceable_count)
        truck_emergency_not_serviceable_before = int(self.truck_emergency_not_serviceable_count)
        island_candidate_before = int(self.island_task_candidate_count)
        island_serviceable_before = int(self.island_task_serviceable_count)
        island_launch_block_before = int(self.island_task_launch_block_count)

        step_island_task_ids = self._current_island_emergency_task_ids()
        if step_island_task_ids:
            self._island_task_ids_seen.update(set(step_island_task_ids))

        reassigned = 0
        servicing_prev = self._servicing_agents()

        # 1) Advance existing truck transit + depot-only replenish service.
        for aid, s in self.state.agents.items():
            if s.kind != AgentKind.TRUCK:
                continue
            self._sync_truck_inventory_fields(s)
            self._advance_truck_transit(aid)
            if s.transit is not None or s.node is None:
                continue

            init_bulk_kg, init_tc_kg = self._truck_replenish_inventory_targets(str(aid))
            needs_refill = bool(self._truck_needs_replenish_for_pending_tasks(s))
            at_depot = bool(int(s.node) == 0)

            if bool(getattr(self.cfg, "truck_replenish_only_at_depot", True)):
                if at_depot and needs_refill:
                    if int(s.truck_replenish_timer) <= 0:
                        s.truck_replenish_timer = int(max(1, int(getattr(self.cfg, "truck_replenish_service_steps", 2))))
                    s.truck_replenish_timer = int(s.truck_replenish_timer) - 1
                    if int(s.truck_replenish_timer) <= 0:
                        s.bulk_inventory_kg_current = float(init_bulk_kg)
                        s.timecritical_inventory_kg_current = float(init_tc_kg)
                        s.truck_needs_replenish_flag = False
                        self._sync_truck_inventory_fields(s)
                        self.truck_replenish_count_total += 1
                        step_truck_replenish_events += 1
                elif at_depot:
                    s.truck_replenish_timer = 0
            elif needs_refill:
                s.bulk_inventory_kg_current = float(init_bulk_kg)
                s.timecritical_inventory_kg_current = float(init_tc_kg)
                s.truck_needs_replenish_flag = False
                self._sync_truck_inventory_fields(s)
                self.truck_replenish_count_total += 1
                step_truck_replenish_events += 1

        # 1.5) Atomic onsite routine capture must run after existing road transit
        # advances but before any new truck command is allowed to start motion.
        # This is intentionally execution-layer authority: planner-only holding
        # was vulnerable to an assist command overwriting the service opportunity.
        pre_motion_capture_services = self._capture_ready_routine_tasks_before_motion()
        if pre_motion_capture_services:
            servicing_prev = self._servicing_agents()

        # 2) Apply actions independently.
        for aid, s in self.state.agents.items():
            if aid in servicing_prev:
                # Agent is unloading cargo: ignore control inputs this step.
                continue
            a = action.get(aid, None)
            if s.kind == AgentKind.TRUCK:
                self._sync_truck_inventory_fields(s)
                truck_action = normalize_truck_step_action(
                    a,
                    in_transit=bool(s.transit is not None),
                    replenish_timer=int(getattr(s, "truck_replenish_timer", 0)),
                )
                if truck_action.ignore:
                    continue
                if truck_action.invalid:
                    rewards[aid] += float(self.cfg.reward_invalid_action)
                    invalid_action_count += 1
                    invalid_action_count_truck += 1
                    reward_invalid_total += float(self.cfg.reward_invalid_action)
                    self._record_invalid_action(
                        aid,
                        a,
                        truck_action.action,
                        validation_layer="environment",
                        reason_code="ACTION_SHAPE_INVALID",
                        reason_detail="environment rejected non-TruckAction",
                        source_code_location="base_env.step:truck_shape",
                    )
                    continue
                a = truck_action.action
                if not isinstance(a, TruckAction):
                    continue

                forced_depot_target: Optional[int] = None
                if self._truck_requires_depot(aid) and s.node is not None and int(s.node) != 0:
                    legal_to_depot = self._decision_neighbors(int(s.node))
                    if legal_to_depot:
                        best_nb = min(
                            legal_to_depot,
                            key=lambda nb: self._decision_shortest_path_distance(
                                int(nb), 0
                            ),
                        )
                        if int(a.target_node) != int(best_nb):
                            # Planner/low-level mismatch during replenish stage is
                            # corrected to depot move to reduce truck invalid friction.
                            forced_depot_target = int(best_nb)

                support_nb = None
                support_mode: Optional[str] = None
                if s.node is not None:
                    neighbor_nodes = list(self._decision_neighbors(int(s.node)))
                    routine_goal_protected = bool(
                        self._truck_routine_goal_support_protected(str(aid), [int(x) for x in neighbor_nodes])
                    )
                    assigned_airborne_hard_recovery = bool(
                        self._truck_has_assigned_airborne_hard_recovery_request(str(aid))
                    )
                    protected_routine_task_id = None
                    if routine_goal_protected:
                        goal_id_protected = self._effective_goals.get(
                            str(aid), self._recommended_goals.get(str(aid), None)
                        )
                        protected_task = (
                            self.state.tasks.get(str(goal_id_protected), None)
                            if goal_id_protected is not None
                            else None
                        )
                        if protected_task is not None and protected_task.kind == TaskKind.NORMAL:
                            protected_routine_task_id = str(protected_task.task_id)
                            self._routine_protection_recent_tasks[protected_routine_task_id] = int(self.state.step_index) + 20
                    if neighbor_nodes:
                        multiround_routine_protected = bool(
                            self._truck_active_multiround_routine_commitment(str(aid))
                            is not None
                        )
                        if self._has_hard_recovery_uav():
                            nb_recovery = self._truck_recovery_support_target(aid, neighbor_nodes)
                            if nb_recovery is not None:
                                if not routine_goal_protected:
                                    support_nb = int(nb_recovery)
                                    support_mode = "recovery"
                                elif assigned_airborne_hard_recovery:
                                    self.routine_near_completion_broken_by_hard_safety_count += 1
                                    support_nb = int(nb_recovery)
                                    support_mode = "recovery"
                                else:
                                    self.routine_near_completion_protected_count += 1
                                    if multiround_routine_protected:
                                        override_ok, override_info = False, {}
                                        self.routine_multiround_support_block_count += 1
                                    else:
                                        override_ok, override_info = self._routine_protection_tc_override_allowed(
                                            str(aid), int(nb_recovery)
                                        )
                                    if override_ok:
                                        self.routine_near_completion_broken_by_tc_override_count += 1
                                        if float(override_info.get("predicted_full_sortie_feasible", 0.0)) > 0.0:
                                            self.routine_near_completion_broken_by_delivery_feasible_tc_override_count += 1
                                        task_id_ov = str(override_info.get("task_id", ""))
                                        if task_id_ov:
                                            self._routine_tc_override_recent_tasks[task_id_ov] = int(self.state.step_index) + 20
                                            self._tc_override_recent_tasks[task_id_ov] = int(self.state.step_index) + 20
                                        support_nb = int(nb_recovery)
                                        support_mode = "recovery"
                                    else:
                                        self.routine_near_completion_recovery_blocked_count += 1
                                        if str(override_info.get("task_id", "")):
                                            self.routine_near_completion_blocked_tc_support_count += 1
                        if (
                            support_nb is None
                            and bool(step_island_task_ids)
                            and (not self._truck_requires_depot(aid))
                        ):
                            # Avoid unconditional truck detours: only engage island
                            # forward-support when this truck is currently tied to
                            # island delivery flow (its own goal is island task or a
                            # docked UAV tracked by this truck is targeting island).
                            has_island_bound_uav = False
                            launch_batt_thr = float(self._uav_launch_min_battery_threshold())
                            for uid, us in self.state.agents.items():
                                if us.kind != AgentKind.UAV or bool(us.crashed):
                                    continue
                                if us.follow_target is None or str(us.follow_target) != str(aid):
                                    continue
                                ugoal = self._effective_goals.get(
                                    str(uid), self._recommended_goals.get(str(uid), None)
                                )
                                if ugoal is None or str(ugoal) not in set(step_island_task_ids):
                                    continue
                                # Only count bind-linked island demand when UAV is sortie-ready.
                                if bool(getattr(us, "uav_needs_reload_flag", False)):
                                    continue
                                if int(getattr(us, "uav_reload_timer", 0)) > 0:
                                    continue
                                if not bool(self._uav_loaded(str(uid))):
                                    continue
                                if float(getattr(us, "battery", 0.0)) + 1e-9 < launch_batt_thr:
                                    continue
                                has_island_bound_uav = True
                                break
                            # Island support should be demand-driven by ready UAV-truck
                            # pairing, otherwise trucks over-detour and hurt completion.
                            # Additional paper policy: truck main duty is NORMAL throughput;
                            # if normal tasks are still reachable/serviceable, do not divert
                            # to island forward-support (except hard recovery branch above).
                            if has_island_bound_uav and (not self._truck_has_reachable_serviceable_normal(str(aid))):
                                nb_island = self._truck_island_forward_support_target(
                                    aid, neighbor_nodes, set(step_island_task_ids)
                                )
                                if nb_island is not None:
                                    if not routine_goal_protected:
                                        support_nb = int(nb_island)
                                        support_mode = "island"
                                    else:
                                        self.routine_near_completion_protected_count += 1
                                        if multiround_routine_protected:
                                            override_ok, override_info = False, {}
                                            self.routine_multiround_support_block_count += 1
                                        else:
                                            override_ok, override_info = self._routine_protection_tc_override_allowed(
                                                str(aid), int(nb_island)
                                            )
                                        if override_ok:
                                            self.routine_near_completion_broken_by_tc_override_count += 1
                                            if float(override_info.get("predicted_full_sortie_feasible", 0.0)) > 0.0:
                                                self.routine_near_completion_broken_by_delivery_feasible_tc_override_count += 1
                                            task_id_ov = str(override_info.get("task_id", ""))
                                            if task_id_ov:
                                                self._routine_tc_override_recent_tasks[task_id_ov] = int(self.state.step_index) + 20
                                                self._tc_override_recent_tasks[task_id_ov] = int(self.state.step_index) + 20
                                            support_nb = int(nb_island)
                                            support_mode = "island"
                                        else:
                                            self.routine_near_completion_support_blocked_count += 1
                                            if str(override_info.get("task_id", "")):
                                                self.routine_near_completion_blocked_tc_support_count += 1

                if (
                    support_nb is None
                    and bool(getattr(self, "_erc_v2_command_gate_enabled", False))
                    and s.node is not None
                    and (not self._truck_requires_depot(aid))
                ):
                    cmd_target = self._erc_v2_truck_command_target(str(aid), "support_uav")
                    if cmd_target is not None and int(cmd_target) in set(int(x) for x in self._decision_neighbors(int(s.node))):
                        support_nb = int(cmd_target)
                        support_mode = "island"
                    rec_target = self._erc_v2_truck_command_target(str(aid), "safety_recovery")
                    if support_nb is None and rec_target is not None and int(rec_target) in set(int(x) for x in self._decision_neighbors(int(s.node))):
                        support_nb = int(rec_target)
                        support_mode = "recovery"

                if support_nb is not None and not self._erc_v2_authorized_truck_support(str(aid), str(support_mode or "")):
                    support_nb = None
                    support_mode = None

                target_node = int(a.target_node)
                if forced_depot_target is not None:
                    target_node = int(forced_depot_target)
                elif (
                    support_nb is not None
                    and (not self._truck_requires_depot(aid))
                ):
                    target_node = int(support_nb)

                # Anti-pingpong guard: if planner asks immediate backtracking and
                # there are other reachable serviceable tasks, switch to a better
                # reachable neighbor instead of oscillating.
                if (
                    forced_depot_target is None
                    and support_nb is None
                    and s.node is not None
                    # Disabled trial: excluding ER-HLNS here reduced seed114
                    # L-A from 95% to 70%; retain the shared safety guard.
                    and True
                ):
                    last_from = self._truck_last_arrived_from.get(str(aid), None)
                    if last_from is not None and int(target_node) == int(last_from):
                        alt_nb, _ = self._truck_best_reachable_service_move(str(aid), avoid_node=int(target_node))
                        if alt_nb is not None:
                            target_node = int(alt_nb)

                ok = self._start_truck_move(aid, int(target_node))
                if (not ok) and (not self._truck_requires_depot(aid)):
                    alt_nb, has_reachable = self._truck_best_reachable_service_move(str(aid), avoid_node=int(target_node))
                    if alt_nb is not None and int(alt_nb) != int(target_node):
                        ok = self._start_truck_move(aid, int(alt_nb))
                        if ok:
                            target_node = int(alt_nb)
                    if (not ok) and alt_nb is None:
                        alt_nb2, has_reachable2 = self._truck_best_reachable_service_move(str(aid), avoid_node=None)
                        has_reachable = bool(has_reachable or has_reachable2)
                        if alt_nb2 is not None:
                            ok = self._start_truck_move(aid, int(alt_nb2))
                            if ok:
                                target_node = int(alt_nb2)
                    # If all goals are currently dead-end/unreachable, allow pause.
                    if (not ok) and (not bool(has_reachable)):
                        continue

                if not ok:
                    rewards[aid] += float(self.cfg.reward_invalid_action)
                    invalid_action_count += 1
                    invalid_action_count_truck += 1
                    reward_invalid_total += float(self.cfg.reward_invalid_action)
                    self._record_invalid_action(
                        aid,
                        a,
                        a,
                        validation_layer="environment",
                        reason_code="ROAD_BECAME_BLOCKED",
                        reason_detail=f"truck could not start move to node {target_node}",
                        source_code_location="base_env.step:truck_start_move",
                    )
                else:
                    moved_dists[aid] = 1.0
                    if self._truck_requires_depot(aid):
                        step_truck_empty_trip_count += 1
                    if support_nb is not None and int(target_node) == int(support_nb):
                        if support_mode == "recovery":
                            self.truck_recovery_support_count_total += 1
                        if bool(step_island_task_ids) and support_mode in {"island", "recovery"}:
                            self.truck_forward_support_count_total += 1
                            if s.node is not None:
                                move_d = float(self.topology.edge_distance(int(s.node), int(target_node)))
                                self.truck_forward_support_distance_total += float(move_d)
                                step_truck_forward_support_distance += float(move_d)
            else:
                uav_action = normalize_uav_step_action(a)
                if uav_action.invalid:
                    rewards[aid] += float(self.cfg.reward_invalid_action)
                    invalid_action_count += 1
                    invalid_action_count_uav += 1
                    reward_invalid_total += float(self.cfg.reward_invalid_action)
                    self._record_invalid_action(
                        aid,
                        a,
                        uav_action.action,
                        validation_layer="environment",
                        reason_code="ACTION_SHAPE_INVALID",
                        reason_detail="environment rejected non-UAVAction",
                        source_code_location="base_env.step:uav_shape",
                    )
                    continue
                a = uav_action.action
                if not isinstance(a, UAVAction):
                    continue
                final_dispatch = validate_action_for_dispatch(self, str(aid), a)
                if not final_dispatch.valid:
                    fallback = final_dispatch.fallback_action
                    if fallback is None:
                        fallback = safe_noop_for_agent_state(s)
                    repair_ok = validate_action_for_dispatch(self, str(aid), fallback).valid
                    if not repair_ok:
                        fallback = safe_noop_for_agent_state(s)
                        repair_ok = validate_action_for_dispatch(self, str(aid), fallback).valid
                    self.pre_dispatch_rejected_count_total += 1
                    self.safe_noop_fallback_count_total += 1
                    if repair_ok:
                        self.pre_dispatch_repair_success_count_total += 1
                    self._record_invalid_action(
                        aid,
                        a,
                        final_dispatch.normalized_action,
                        validation_layer="pre_dispatch",
                        reason_code=str(final_dispatch.reason_code or "UNKNOWN_INVALID_REASON"),
                        reason_detail=str(final_dispatch.reason_detail),
                        local_repair_attempted=True,
                        local_repair_succeeded=bool(repair_ok),
                        fallback_action=fallback,
                        source_code_location=str(final_dispatch.source_code_location or "base_env.step:uav_final_dispatch"),
                    )
                    a = fallback if isinstance(fallback, UAVAction) else UAVAction(vx=0.0, vy=0.0)
                self._uav_motion_old_xy[str(aid)] = (
                    tuple(float(v) for v in s.pos_xy)
                    if s.pos_xy is not None
                    else self._node_xy(int(s.node or 0))
                )
                invalid, moved_dist, new_bind, hw, rain, queue_waiting, sortie_limited = self._apply_uav_action(aid, a)
                if invalid:
                    rewards[aid] += float(self.cfg.reward_invalid_action)
                    invalid_action_count += 1
                    invalid_action_count_uav += 1
                    reward_invalid_total += float(self.cfg.reward_invalid_action)
                    reason = str(self._last_uav_invalid_reason.get(str(aid), "UNKNOWN_INVALID_REASON"))
                    self._record_invalid_action(
                        aid,
                        a,
                        a,
                        validation_layer="environment",
                        reason_code=reason if reason else "UNKNOWN_INVALID_REASON",
                        reason_detail="environment _apply_uav_action rejected action",
                        source_code_location="base_env.step:uav_apply",
                    )
                if queue_waiting:
                    step_queue_wait_steps += 1
                if sortie_limited:
                    step_sortie_limit_hits += 1
                    if self._uav_mark_forced_recovery(aid):
                        step_forced_rth_count += 1
                if new_bind:
                    step_follow_bind_count += 1
                    batt_before = float(pre_step_battery.get(aid, 1.0))
                    if batt_before < float(self.cfg.docking_reward_battery_threshold):
                        dock_bonus = float(self.cfg.reward_docking_low_battery)
                        rewards[aid] += dock_bonus
                        reward_docking_total += dock_bonus
                    if (
                        batt_before < float(self.cfg.docking_reward_battery_threshold)
                        or (not self._uav_loaded(aid))
                    ):
                        if self._uav_mark_forced_recovery(aid):
                            step_forced_rth_count += 1
                moved_dists[aid] = moved_dist
                moved_headwind[aid] = float(hw)
                moved_rain[aid] = float(rain)
                goal_id_uav = self._effective_goals.get(
                    str(aid), self._recommended_goals.get(str(aid), None)
                )
                goal_task_uav = self.state.tasks.get(str(goal_id_uav)) if goal_id_uav is not None else None
                if (
                    moved_dist > 1e-6
                    and s.follow_target is None
                    and (not self._uav_loaded(aid))
                ):
                    step_uav_empty_flight_count += 1
                    self.uav_empty_flight_count_total += 1
                    # Penalize empty-payload emergency chasing to avoid reward hacking via empty flights.
                    if (
                        goal_task_uav is not None
                        and goal_task_uav.kind == TaskKind.EMERGENCY
                        and goal_task_uav.status == TaskStatus.PENDING
                        and (not bool(self._uav_recovery_required(str(aid))))
                    ):
                        # This is a performance/supply diagnostic, not an action
                        # validator rejection. Keep it in uav_empty_flight counters
                        # so environment_invalid_action_count only measures actions
                        # actually rejected by environment validation.
                        pass

        # 2.5) Start unloading service for agents that reached task points.
        started_services = pre_motion_capture_services + self._start_service_for_arrived_agents()
        if started_services:
            now_step = int(self.state.step_index)
            for _, svc_task in started_services:
                tid = str(svc_task.task_id)
                if (
                    svc_task.kind == TaskKind.NORMAL
                    and int(self._routine_protection_recent_tasks.get(tid, -1)) >= now_step
                ):
                    self.routine_near_completion_followed_by_service_start_count += 1
                    self._routine_protection_recent_tasks.pop(tid, None)
        servicing_now = self._servicing_agents()
        service_started_by_uav_step = sum(
            1
            for aid, _ in started_services
            if self.state.agents.get(str(aid)) is not None
            and self.state.agents[str(aid)].kind == AgentKind.UAV
        )
        service_started_by_truck_step = sum(
            1
            for aid, _ in started_services
            if self.state.agents.get(str(aid)) is not None
            and self.state.agents[str(aid)].kind == AgentKind.TRUCK
        )
        if started_services:
            pickup_bonus = float(getattr(self.cfg, "reward_pickup", 0.0))
            if abs(pickup_bonus) > 1e-12:
                for aid, _ in started_services:
                    if aid in rewards:
                        rewards[aid] += pickup_bonus
                        reward_pickup_total += pickup_bonus
        for aid in servicing_now:
            # No step penalty during unloading.
            rewards[aid] -= float(self.cfg.reward_step_penalty)
            reward_step_total -= float(self.cfg.reward_step_penalty)

        # 3) Follow sync + charging + crash.
        crashed_agents = []
        for aid, s in self.state.agents.items():
            if s.kind != AgentKind.UAV:
                continue
            if aid in servicing_now:
                # During unloading: no energy drain and no charging bookkeeping.
                s.vel_xy = (0.0, 0.0)
                continue
            # Optional env-side autonomous departure is disabled by default for
            # paper safety runs; policy must issue explicit takeoff.
            auto_depart_enabled = bool(getattr(self.cfg, "uav_auto_depart_when_ready", False))
            goal_id_loop = self._effective_goals.get(str(aid), self._recommended_goals.get(str(aid), None))
            goal_task_loop = self.state.tasks.get(str(goal_id_loop)) if goal_id_loop is not None else None
            if (
                auto_depart_enabled
                and s.follow_target is not None
                and self._uav_loaded(aid)
                and int(s.replenish_timer) <= 0
                and int(getattr(s, "uav_reload_timer", 0)) <= 0
                and goal_task_loop is not None
                and goal_task_loop.kind == TaskKind.EMERGENCY
                and goal_task_loop.status == TaskStatus.PENDING
                and int(self._uav_post_bind_dwell_remaining.get(str(aid), 0)) <= 0
            ):
                launch_ok, launch_reason, force_recovery = self._uav_launch_gate_check(
                    str(aid), task=goal_task_loop
                )
                if launch_ok and (not force_recovery):
                    launch_truck_id = str(s.follow_target or "")
                    if launch_ok and self._commit_uav_delivery_launch(
                        str(aid),
                        goal_task_loop,
                        str(launch_reason),
                        launch_truck_id=launch_truck_id,
                    ):
                        step_forced_takeoff_full += 1
                        self._uav_forced_rth_latch[aid] = False
            low_evt_thr = float(np.clip(getattr(self.cfg, "uav_low_battery_goal_lock_threshold", 0.35), 0.0, 1.0))
            force_recover_thr = float(np.clip(getattr(self.cfg, "uav_low_battery_force_recover_threshold", 0.25), 0.0, 1.0))
            if float(s.battery) < low_evt_thr and not self._uav_low_battery_flag.get(aid, False):
                self._uav_low_battery_flag[aid] = True
                step_low_battery_events += 1
            if float(s.battery) <= force_recover_thr:
                if self._uav_mark_forced_recovery(aid, reason="low_battery"):
                    step_forced_rth_count += 1

            if s.follow_target is not None:
                step_follow_steps += 1
            charged = self._sync_follow_and_charge(aid)
            step_follow_charge_energy += float(charged)

            if (
                self._uav_low_battery_flag.get(aid, False)
                and s.follow_target is not None
                and charged > 1e-9
            ):
                self._uav_low_battery_flag[aid] = False
                step_low_battery_return_success += 1

            if self._uav_forced_rth_latch.get(aid, False):
                # Release only when the actual assigned sortie is feasible;
                # do not require an unrelated fixed SOC such as 0.78.
                sortie_ready = False
                if s.follow_target is not None and self._uav_loaded(aid):
                    sortie_ready, _, _ = self._uav_launch_gate_check(aid, task=goal_task_loop, count_reject=False)
                ready_release = bool(
                    sortie_ready
                    and int(getattr(s, "uav_reload_timer", 0)) <= 0
                    and (not bool(getattr(s, "uav_needs_reload_flag", False)))
                )
                if ready_release:
                    self._uav_forced_rth_latch[aid] = False

            crashed = self._update_uav_energy_and_crash(
                aid, moved_dists[aid], moved_headwind[aid], moved_rain[aid]
            )
            wind_failure_risk = self._uav_wind_failure_risk(aid)
            low_soc_failure_risk = self._uav_low_soc_failure_risk(aid)
            step_wind_failure_risk_accum += float(wind_failure_risk)
            failure_risk = float(np.clip(wind_failure_risk + low_soc_failure_risk, 0.0, 1.0))
            if (not crashed) and failure_risk > 0.0 and self.rng.uniform() < failure_risk:
                s.battery = 0.0
                s.crashed = True
                s.follow_target = None
                s.sortie_distance_m = 0.0
                crashed = True
                step_wind_failure_event_count += 1
            batt_after = float(s.battery)
            batt_before_energy = float(pre_step_battery.get(aid, batt_after))
            # Charging may happen before energy update; consumption uses positive drop only.
            if batt_before_energy > batt_after:
                step_uav_energy_used += float(batt_before_energy - batt_after)
            pre_step_battery[aid] = batt_after
            if crashed:
                rewards[aid] += float(self.cfg.uav_crash_penalty)
                reward_crash_total += float(self.cfg.uav_crash_penalty)
                crashed_agents.append(aid)
                self._uav_low_battery_flag[aid] = False
                self._uav_forced_rth_latch[aid] = False
                self._uav_post_bind_dwell_remaining[aid] = 0
                self._uav_bind_commit_target[aid] = None
                self._uav_bind_commit_until_step[aid] = -1
                step_crash_count += 1
                step_battery_depletion_count += 1

        # 4) Idle penalty only if agent has active assigned task.
        for aid, s in self.state.agents.items():
            if aid in servicing_now:
                continue
            if self._pbrs_target_task(aid) is None:
                continue
            stationary = moved_dists[aid] <= 1e-6
            if s.kind == AgentKind.TRUCK and s.transit is not None:
                stationary = False
            if s.kind == AgentKind.UAV and s.follow_target is not None:
                stationary = False
            if stationary:
                rewards[aid] += float(self.cfg.reward_idle_with_task)
                reward_idle_total += float(self.cfg.reward_idle_with_task)

        # 5) Delivery/timeout resolution.
        delivered, timed_out, transfers = self._advance_service_and_timeouts()
        delivered_normal = 0
        delivered_emergency = 0
        delivered_by_uav_step = 0
        delivered_emergency_by_uav_step = 0
        delivered_by_truck_step = 0
        for aid, task, transfer in transfers:
            if transfer <= 0.0:
                continue
            task_units = float(max(self._task_demand_kg(task), 1e-6))
            if task.kind == TaskKind.NORMAL:
                bonus = float(self.cfg.reward_delivery_normal) * float(transfer / task_units)
            else:
                bonus = float(self.cfg.reward_delivery_emergency) * float(transfer / task_units)
                agent_state = self.state.agents.get(str(aid))
                if agent_state is not None and agent_state.kind == AgentKind.UAV:
                    uav_bonus = float(self.cfg.reward_uav_emergency_delivery_bonus) * float(
                        transfer / task_units
                    )
                    bonus += uav_bonus
                    reward_uav_emergency_bonus_total += float(uav_bonus)
            rewards[aid] += float(bonus)
            reward_delivery_total += float(bonus)
        for _, task in delivered:
            if task.kind == TaskKind.NORMAL:
                delivered_normal += 1
                tid = str(task.task_id)
                if int(self._routine_protection_recent_tasks.get(tid, -1)) >= int(self.state.step_index):
                    self.routine_near_completion_followed_by_completion_count += 1
                    self._routine_protection_recent_tasks.pop(tid, None)
            else:
                delivered_emergency += 1
                tid = str(task.task_id)
                if int(self._routine_tc_override_recent_tasks.get(tid, -1)) >= int(self.state.step_index):
                    self.routine_near_completion_tc_override_to_delivery_count += 1
                    self._routine_tc_override_recent_tasks.pop(tid, None)
                if int(self._tc_override_recent_tasks.get(tid, -1)) >= int(self.state.step_index):
                    self.tc_override_to_delivery_count += 1
                    self.tc_override_actual_delivery_count += 1
                    self._tc_override_recent_tasks.pop(tid, None)
                    for row in reversed(self._tc_override_trace_rows):
                        if str(row.get("tc_task_id", "")) == tid and int(row.get("actual_delivery_after", 0)) == 0:
                            row["actual_delivery_after"] = 1
                            break
            if task.kind == TaskKind.EMERGENCY and str(task.task_id) in self._island_task_ids_seen:
                self.island_task_completed_count_total += 1
            deliver_aid = str(task.delivered_by) if task.delivered_by is not None else ""
            deliver_agent = self.state.agents.get(deliver_aid, None)
            if deliver_agent is None:
                continue
            if deliver_agent.kind == AgentKind.UAV:
                delivered_by_uav_step += 1
                if task.kind == TaskKind.EMERGENCY:
                    delivered_emergency_by_uav_step += 1
                    if bool(self._uav_sortie_relaxed_latch.get(str(deliver_aid), False)):
                        self.relaxed_delivery_completed_count_total += 1
                    if str(task.task_id) in self._island_task_ids_seen:
                        self.uav_island_delivery_count_total += 1
            elif deliver_agent.kind == AgentKind.TRUCK:
                delivered_by_truck_step += 1

        if delivered:
            shared_total_per_delivery = float(getattr(self.cfg, "reward_delivery_shared", 0.0))
            if abs(shared_total_per_delivery) > 1e-12:
                share_each = float(shared_total_per_delivery / max(len(rewards), 1))
                for _ in delivered:
                    for aid in rewards:
                        rewards[aid] += share_each
                        reward_delivery_shared_total += share_each
        failed_normal = 0
        failed_emergency = 0
        for task in timed_out:
            if task.kind == TaskKind.NORMAL:
                penalty = float(self.cfg.penalty_timeout_normal)
                failed_normal += 1
            else:
                penalty = float(self.cfg.penalty_timeout_emergency)
                failed_emergency += 1
            if task.assigned_to is not None and str(task.assigned_to) in rewards:
                rewards[str(task.assigned_to)] += penalty
                reward_timeout_total += float(penalty)
            else:
                share = penalty / max(len(rewards), 1)
                for aid in rewards:
                    rewards[aid] += share
                reward_timeout_total += float(penalty)

        # 6) PBRS hook with strict target-consistency lock.
        if bool(self.cfg.use_pbrs):
            gamma = float(self.cfg.pbrs_gamma)
            norm_m = float(max(self.cfg.pbrs_distance_norm_m, 1e-6))
            for aid in rewards:
                current_task = self._pbrs_target_task(aid)
                if current_task is None:
                    self._pbrs_lock[aid] = (None, None)
                    continue
                cur_tid = str(current_task.task_id)
                cur_dist = self._agent_distance_to_task(aid, current_task)
                lock_tid, lock_dist = self._pbrs_lock.get(aid, (None, None))
                if lock_tid is None or lock_dist is None:
                    self._pbrs_lock[aid] = (cur_tid, float(cur_dist))
                    continue
                if str(lock_tid) != str(cur_tid):
                    # Hard reset on target switch: no shaping reward on switch step.
                    step_pbrs_switch_count += 1
                    self._pbrs_lock[aid] = (cur_tid, float(cur_dist))
                    continue
                phi_prev = -float(lock_dist) / norm_m
                phi_new = -float(cur_dist) / norm_m
                shaping = float(self.cfg.pbrs_scale) * (gamma * phi_new - phi_prev)
                rewards[aid] += shaping
                reward_pbrs_total += float(shaping)
                self._pbrs_lock[aid] = (cur_tid, float(cur_dist))

        if hasattr(self.hazards, "set_forced_island_edges"):
            forced_edges = set(self._forced_island_edge_keys) if bool(getattr(self.cfg, "forced_island_lock_edges", True)) else set()
            self.hazards.set_forced_island_edges(forced_edges)
        elif hasattr(self.hazards, "set_nonreopen_edges"):
            forced_nonreopen = set(self._forced_island_edge_keys) if bool(getattr(self.cfg, "forced_island_lock_edges", True)) else set()
            self.hazards.set_nonreopen_edges(forced_nonreopen)
        rain_mean, wind_mean, blocked_ratio, epicenter = self.hazards.step()
        # Keep forced island road cuts active when lock is enabled.
        self._enforce_forced_island_blocking()
        if hasattr(self.hazards, "_refresh_blockage_partition_stats"):
            self.hazards._refresh_blockage_partition_stats()
        blocked_ratio = float(getattr(self.hazards, "blocked_ratio_total", self.topology.blocked_ratio()))
        if hasattr(self.hazards, "blocked_edge_count"):
            self.hazards.blocked_edge_count = int(len(self.topology.blocked_edges))
        self.state.hazard.rainfall_mean = rain_mean
        self.state.hazard.wind_mean = wind_mean
        self.state.hazard.blocked_ratio = blocked_ratio
        self.state.hazard.blocked_ratio_stochastic = float(getattr(self.hazards, "blocked_ratio_stochastic", 0.0))
        self.state.hazard.blocked_ratio_forced_island = float(getattr(self.hazards, "blocked_ratio_forced_island", 0.0))
        self.state.hazard.blocked_ratio_total = float(getattr(self.hazards, "blocked_ratio_total", blocked_ratio))
        self.state.hazard.blockage_target_ratio = float(getattr(self.hazards, "blockage_target_ratio_step", 0.0))
        self.state.hazard.blockage_gap = float(getattr(self.hazards, "blockage_gap_step", 0.0))
        self.state.hazard.blockage_global_gate = float(getattr(self.hazards, "blockage_global_gate_step", 0.0))
        self.state.hazard.epicenter_node = epicenter
        risk_spike = self.state.hazard.blocked_ratio >= self.cfg.risk_spike_threshold
        self.state.hazard.risk_spike = risk_spike
        self._update_comm_blocked()
        self.comm_blackout_agent_observation_count_total = int(
            getattr(self, "comm_blackout_agent_observation_count_total", 0)
        ) + int(len(self.comm_blocked))
        self.comm_blackout_agent_blocked_count_total = int(
            getattr(self, "comm_blackout_agent_blocked_count_total", 0)
        ) + int(
            sum(1 for value in self.comm_blocked.values() if bool(value))
        )
        self.comm_blackout_physical_zone_count_total = int(
            getattr(self, "comm_blackout_physical_zone_count_total", 0)
        ) + int(
            sum(
                1
                for reason in self._comm_block_reason.values()
                if str(reason) in {"physical_zone", "physical_and_goal_zone"}
            )
        )
        self.comm_blackout_goal_zone_count_total = int(
            getattr(self, "comm_blackout_goal_zone_count_total", 0)
        ) + int(
            sum(
                1
                for reason in self._comm_block_reason.values()
                if str(reason) in {"goal_zone", "physical_and_goal_zone"}
            )
        )

        # Stage-C shared-road cognition update:
        # UAV/truck scouts publish observed blocked/cleared edges into shared map.
        self._shared_awareness_step_update()

        # 7) UAV scouting reward: first-time blocked-edge discovery within monitor radius.
        if self._uav_scouting_enabled():
            for aid, s in self.state.agents.items():
                if s.kind != AgentKind.UAV or s.crashed:
                    continue
                visible_blocked = self._uav_visible_blocked_edges(
                    aid, radius_m=float(self.cfg.uav_monitor_radius_m)
                )
                if not visible_blocked:
                    continue
                new_count = 0
                for edge_key in visible_blocked:
                    if edge_key in self._uav_discovered_blocked_edges:
                        continue
                    self._uav_discovered_blocked_edges.add(edge_key)
                    new_count += 1
                if new_count > 0:
                    bonus = float(self.cfg.reward_uav_discover_blocked_edge) * float(new_count)
                    rewards[aid] += bonus
                    reward_discover_total += bonus
                    step_uav_discovered_blocked += int(new_count)
                    self._uav_discovered_blocked_total += int(new_count)

        tasks_terminal = all(
            t.status in (TaskStatus.DELIVERED, TaskStatus.FAILED)
            for t in self.state.tasks.values()
        )
        uav_settled = self._uav_all_settled_for_termination()
        if tasks_terminal and uav_settled:
            self.state.done = True
        if self.state.step_index >= self.cfg.max_steps:
            self.state.done = True

        self.follow_bind_count_total += int(step_follow_bind_count)
        self.follow_steps_total += int(step_follow_steps)
        self.follow_charge_energy_total += float(step_follow_charge_energy)
        self.low_battery_events_total += int(step_low_battery_events)
        self.low_battery_return_success_total += int(step_low_battery_return_success)
        self.uav_energy_used_total += float(step_uav_energy_used)
        self.crash_count_total += int(step_crash_count)
        self.battery_depletion_count_total += int(step_battery_depletion_count)
        self.invalid_action_count_total += int(invalid_action_count)
        self.invalid_action_count_uav_total += int(invalid_action_count_uav)
        self.invalid_action_count_truck_total += int(invalid_action_count_truck)
        self.environment_invalid_action_count_total += int(invalid_action_count)
        self.forced_rth_count_total += int(step_forced_rth_count)
        self.queue_wait_steps_total += int(step_queue_wait_steps)
        self._pbrs_switch_total += int(step_pbrs_switch_count)
        self.uav_delivered_tasks_total += int(delivered_by_uav_step)
        self.uav_delivered_emergency_total += int(delivered_emergency_by_uav_step)
        self.uav_delivery_count_total += int(delivered_by_uav_step)
        self.truck_delivered_tasks_total += int(delivered_by_truck_step)
        self.truck_empty_trip_count_total += int(step_truck_empty_trip_count)
        self.sortie_limit_hit_total += int(step_sortie_limit_hits)
        self.wind_failure_event_total += int(step_wind_failure_event_count)
        self.wind_failure_risk_accum_total += float(step_wind_failure_risk_accum)
        uav_moves = [
            aid
            for aid, st in self.state.agents.items()
            if st.kind == AgentKind.UAV and moved_dists.get(aid, 0.0) > 1e-6
        ]
        mean_headwind = float(
            np.mean([moved_headwind[aid] for aid in uav_moves]) if uav_moves else 0.0
        )
        mean_rain = float(np.mean([moved_rain[aid] for aid in uav_moves]) if uav_moves else 0.0)
        node_winds = [float(h.wind) for h in self.hazards.node_hazard.values()]
        node_rains = [float(h.rain) for h in self.hazards.node_hazard.values()]
        wind_severity_p95_mps = float(np.percentile(node_winds, 95)) if node_winds else 0.0
        rain_severity_p95_mmh = float(np.percentile(node_rains, 95)) if node_rains else 0.0
        uav_following_count = 0
        uav_follow_with_goal_count = 0
        uav_follow_near_goal_count = 0
        uav_follow_far_goal_count = 0
        for aid, s in self.state.agents.items():
            if s.kind != AgentKind.UAV:
                continue
            tgt = self._pbrs_target_task(aid)
            if tgt is not None:
                step_uav_goal_task_count += 1
                if s.pos_xy is not None:
                    tn = self.topology.nodes[int(tgt.demand_node)]
                    d = float(np.hypot(float(s.pos_xy[0]) - tn.x, float(s.pos_xy[1]) - tn.y))
                    if d <= float(self.cfg.uav_monitor_radius_m):
                        step_uav_terminal_zone_count += 1
                    if d <= float(self.cfg.uav_delivery_radius_m):
                        step_uav_delivery_zone_count += 1
            if s.follow_target is None:
                continue
            uav_following_count += 1
            if tgt is None or s.pos_xy is None:
                continue
            uav_follow_with_goal_count += 1
            tn = self.topology.nodes[int(tgt.demand_node)]
            d = float(np.hypot(float(s.pos_xy[0]) - tn.x, float(s.pos_xy[1]) - tn.y))
            if d <= float(self.cfg.uav_monitor_radius_m):
                uav_follow_near_goal_count += 1
            else:
                uav_follow_far_goal_count += 1

        island_pending_tasks = [
            t
            for t in self.state.tasks.values()
            if t.kind == TaskKind.EMERGENCY
            and t.status == TaskStatus.PENDING
            and str(t.task_id) in set(step_island_task_ids)
        ]
        island_candidate_count_step = int(len(island_pending_tasks))
        island_serviceable_count_step = 0
        for task_island in island_pending_tasks:
            serviceable_any = False
            for uid, us in self.state.agents.items():
                if us.kind != AgentKind.UAV or bool(getattr(us, "crashed", False)):
                    continue
                if bool(getattr(us, "uav_needs_reload_flag", False)) or (not bool(self._uav_loaded(str(uid)))):
                    continue
                if us.follow_target is not None:
                    ok_lp, reason_lp, _ = self._uav_launch_gate_check(str(uid), task=task_island, count_reject=False)
                    if bool(ok_lp) and (str(reason_lp) == "direct_safe" or str(reason_lp).startswith("rendezvous_safe")):
                        serviceable_any = True
                        break
                else:
                    d_go = float(self._agent_distance_to_task(str(uid), task_island))
                    if not np.isfinite(d_go):
                        continue
                    cur_xy = us.pos_xy if us.pos_xy is not None else self._node_xy(int(us.node or 0))
                    txy = self._node_xy(int(task_island.demand_node))
                    _, d_back = self._nearest_truck_from_xy(txy)
                    if not np.isfinite(d_back):
                        continue
                    req_go = float(self._uav_energy_cost_fraction(str(uid), d_go, cur_xy))
                    req_back = float(self._uav_energy_cost_fraction(str(uid), float(d_back), txy))
                    reserve = float(np.clip(getattr(self.cfg, "uav_emergency_reserve_fraction", 0.20), 0.0, 1.0))
                    rendez_margin = float(np.clip(getattr(self.cfg, "uav_rendezvous_margin_fraction", 0.10), 0.0, 1.0))
                    if float(getattr(us, "battery", 0.0)) + 1e-9 >= float(req_go + req_back + reserve + rendez_margin):
                        serviceable_any = True
                        break
            if serviceable_any:
                island_serviceable_count_step += 1

        island_launch_block_count_step = int(max(island_candidate_count_step - island_serviceable_count_step, 0))
        self.island_task_candidate_count += int(island_candidate_count_step)
        self.island_task_serviceable_count += int(island_serviceable_count_step)
        self.island_task_launch_block_count += int(island_launch_block_count_step)

        step_uav_reload_events = int(self.uav_reload_count_total - reload_count_before)
        step_truck_replenish_events = int(self.truck_replenish_count_total - replenish_count_before)
        step_normal_blocked_supply = int(self.normal_tasks_blocked_by_supply_count - normal_block_before)
        step_emergency_blocked_supply = int(self.emergency_tasks_blocked_by_supply_count - emergency_block_before)
        step_uav_recharge_events = int(self.uav_recharge_count_total - recharge_count_before)
        step_uav_safe_launch = int(self.uav_safe_launch_count_total - safe_launch_before)
        step_uav_launch = int(self.uav_launch_count_total - launch_count_before)
        if step_uav_launch > 0:
            active_override_tasks = [
                tid
                for tid, expire in list(self._routine_tc_override_recent_tasks.items())
                if int(expire) >= int(self.state.step_index)
            ]
            self.routine_near_completion_tc_override_to_launch_count += int(
                min(int(step_uav_launch), len(active_override_tasks))
            )
            active_delivery_override_tasks = [
                tid
                for tid, expire in list(self._tc_override_recent_tasks.items())
                if int(expire) >= int(self.state.step_index)
            ]
            launch_delta = int(min(int(step_uav_launch), len(active_delivery_override_tasks)))
            self.tc_override_to_launch_count += int(launch_delta)
            self.tc_override_actual_launch_count += int(launch_delta)
            if launch_delta > 0:
                marked = 0
                for row in reversed(self._tc_override_trace_rows):
                    if int(row.get("override_allowed", 0)) == 1 and int(row.get("actual_launch_after", 0)) == 0:
                        row["actual_launch_after"] = 1
                        marked += 1
                        if marked >= launch_delta:
                            break
        step_uav_unsafe_launch_attempt = int(self.uav_unsafe_launch_attempt_count_total - unsafe_launch_attempt_before)
        step_uav_unsafe_launch_block = int(self.uav_unsafe_launch_block_count_total - unsafe_launch_before)
        step_uav_low_battery_illegal_launch = int(
            self.uav_low_battery_illegal_launch_count_total - low_batt_illegal_launch_before
        )
        step_uav_forced_recovery = int(self.uav_forced_recovery_count_total - forced_recovery_before)
        if step_uav_forced_recovery > 0:
            active_delivery_override_tasks = [
                tid
                for tid, expire in list(self._tc_override_recent_tasks.items())
                if int(expire) >= int(self.state.step_index)
            ]
            forced_delta = int(min(int(step_uav_forced_recovery), len(active_delivery_override_tasks)))
            self.tc_override_to_forced_recovery_count += int(forced_delta)
            self.tc_override_feasibility_mismatch_count += int(forced_delta)
            if forced_delta > 0:
                ttl = int(max(getattr(self.cfg, "hrl_tc_override_reject_cache_ttl_steps", 20), 0))
                marked = 0
                for row in reversed(self._tc_override_trace_rows):
                    if int(row.get("override_allowed", 0)) == 1 and int(row.get("actual_forced_recovery_after", 0)) == 0:
                        row["actual_forced_recovery_after"] = 1
                        self._tc_override_recent_reject[
                            (str(row.get("uav_id", "")), str(row.get("tc_task_id", "")))
                        ] = int(self.state.step_index) + ttl
                        marked += 1
                        if marked >= forced_delta:
                            break
        step_uav_forced_recovery_low_batt = int(
            self.uav_forced_recovery_due_to_low_battery_count_total - forced_recovery_low_batt_before
        )
        step_uav_rendezvous_success = int(self.uav_rendezvous_success_count_total - rendezvous_success_before)
        step_uav_rendezvous_fail = int(self.uav_rendezvous_fail_count_total - rendezvous_fail_before)
        step_truck_recovery_support = int(self.truck_recovery_support_count_total - truck_recovery_support_before)
        step_truck_forward_support = int(self.truck_forward_support_count_total - truck_forward_support_before)
        step_truck_forward_support_distance = float(
            self.truck_forward_support_distance_total - truck_forward_support_dist_before
        )
        step_island_task_completed = int(self.island_task_completed_count_total - island_completed_before)
        step_uav_island_delivery = int(self.uav_island_delivery_count_total - island_delivery_before)
        step_uav_island_recovery_success = int(
            self.uav_island_recovery_success_count_total - island_recovery_before
        )

        step_uav_task_reject_below_launch_min = int(self.uav_task_reject_below_launch_min_count - uav_reject_below_before)
        step_uav_task_reject_not_loaded = int(self.uav_task_reject_not_loaded_count - uav_reject_not_loaded_before)
        step_uav_task_reject_recovery_margin = int(self.uav_task_reject_recovery_margin_count - uav_reject_margin_before)
        step_uav_task_reject_horizon = int(self.uav_task_reject_horizon_count - uav_reject_horizon_before)
        step_uav_task_reject_comm_block = int(self.uav_task_reject_comm_block_count - uav_reject_comm_before)
        step_uav_task_reject_corridor = int(self.uav_task_reject_corridor_count - uav_reject_corridor_before)
        step_uav_launch_direct_safe = int(self.uav_launch_direct_safe_count - uav_launch_direct_before)
        step_uav_launch_rendezvous_safe = int(self.uav_launch_rendezvous_safe_count - uav_launch_rendezvous_before)
        step_uav_launch_rendezvous_safe_relaxed = int(self.uav_launch_rendezvous_safe_relaxed_count - uav_launch_rendezvous_relaxed_before)
        step_uav_launch_block_unsafe = int(self.uav_launch_block_unsafe_count - uav_launch_block_unsafe_before)
        step_uav_launch_gate_enter = int(self.uav_launch_gate_enter_count - uav_launch_gate_enter_before)
        step_uav_launch_gate_direct = int(self.uav_launch_gate_direct_safe_count - uav_launch_gate_direct_before)
        step_uav_launch_gate_rendezvous = int(self.uav_launch_gate_rendezvous_safe_count - uav_launch_gate_rendezvous_before)
        step_uav_launch_gate_rendezvous_relaxed = int(self.uav_launch_gate_rendezvous_safe_relaxed_count - uav_launch_gate_rendezvous_relaxed_before)
        step_uav_launch_gate_block_below = int(self.uav_launch_gate_block_below_launch_min_count - uav_launch_gate_block_below_before)
        step_uav_launch_gate_block_margin = int(self.uav_launch_gate_block_recovery_margin_count - uav_launch_gate_block_margin_before)
        step_uav_launch_gate_block_corridor = int(self.uav_launch_gate_block_corridor_count - uav_launch_gate_block_corridor_before)
        step_uav_launch_gate_block_other = int(self.uav_launch_gate_block_other_count - uav_launch_gate_block_other_before)
        step_truck_emergency_blocked_by_normal_guard = int(self.truck_emergency_blocked_by_normal_guard_count - truck_emergency_block_guard_before)
        step_truck_emergency_relief_override = int(self.truck_emergency_relief_override_count - truck_emergency_relief_override_before)
        step_truck_emergency_serviceable = int(self.truck_emergency_serviceable_count - truck_emergency_serviceable_before)
        step_truck_emergency_not_serviceable = int(self.truck_emergency_not_serviceable_count - truck_emergency_not_serviceable_before)
        step_island_task_candidate = int(self.island_task_candidate_count - island_candidate_before)
        step_island_task_serviceable = int(self.island_task_serviceable_count - island_serviceable_before)
        step_island_task_launch_block = int(self.island_task_launch_block_count - island_launch_block_before)

        if step_uav_rendezvous_success > 0 and bool(step_island_task_ids):
            self.uav_island_recovery_success_count_total += int(step_uav_rendezvous_success)
            step_uav_island_recovery_success += int(step_uav_rendezvous_success)

        truck_states = [st for st in self.state.agents.values() if st.kind == AgentKind.TRUCK]
        uav_states = [st for st in self.state.agents.values() if st.kind == AgentKind.UAV]
        truck_normal_supply_units_total = int(sum(int(getattr(st, "normal_supply_units", 0)) for st in truck_states))
        truck_emergency_supply_units_total = int(sum(int(getattr(st, "emergency_supply_units", 0)) for st in truck_states))
        truck_inventory_kg_current_mean = float(
            np.mean([float(getattr(st, "truck_inventory_kg_current", 0.0)) for st in truck_states])
        ) if truck_states else 0.0
        uav_loaded_fraction = float(
            np.mean(
                [
                    1.0 if (
                        int(getattr(st, "carried_emergency_units", 0)) >= 1
                        and float(getattr(st, "payload_kg_current", 0.0)) >= float(self.cfg.emergency_task_demand_kg) - 1e-9
                    ) else 0.0
                    for st in uav_states
                ]
            )
        ) if uav_states else 0.0
        uav_dead_count = int(sum(1 for st in uav_states if bool(getattr(st, "crashed", False))))
        uav_survival_rate = float(1.0 - (uav_dead_count / max(len(uav_states), 1)))
        uav_crash_free_rate = float(1.0 - (self.crash_count_total / max(len(uav_states), 1)))
        uav_battery_depletion_free_rate = float(1.0 - (self.battery_depletion_count_total / max(len(uav_states), 1)))
        # Raw safety view counts hard-guard rescues as terminal-failure equivalents,
        # to separate policy capability from safety guard intervention.
        uav_raw_failure_equivalent = int(self.crash_count_total + self.uav_terminal_battery_rescue_count_total)
        uav_survival_rate_raw = float(1.0 - (uav_raw_failure_equivalent / max(len(uav_states), 1)))
        uav_crash_free_rate_raw = float(1.0 - (uav_raw_failure_equivalent / max(len(uav_states), 1)))
        uav_launch_battery_fraction_mean = float(
            self.uav_launch_battery_fraction_sum / max(int(self.uav_launch_count_total), 1)
        ) if int(self.uav_launch_count_total) > 0 else 0.0
        uav_launch_battery_fraction_min = float(self.uav_launch_battery_fraction_min) if int(self.uav_launch_count_total) > 0 else 0.0
        island_task_count_total = int(len(self._island_task_ids_seen))
        truck_distance_total_m = float(
            sum(float(getattr(st, "lifetime_distance_m", 0.0)) for st in truck_states)
        )
        uav_distance_total_m = float(
            sum(float(getattr(st, "lifetime_distance_m", 0.0)) for st in uav_states)
        )
        fleet_distance_total_m = float(truck_distance_total_m + uav_distance_total_m)
        delivered_steps = [
            int(getattr(t, "delivered_step", -1))
            for t in self.state.tasks.values()
            if t.status == TaskStatus.DELIVERED and getattr(t, "delivered_step", None) is not None
        ]
        terminal_steps: List[int] = []
        for t in self.state.tasks.values():
            if t.status == TaskStatus.DELIVERED and getattr(t, "delivered_step", None) is not None:
                terminal_steps.append(int(t.delivered_step))
            elif t.status == TaskStatus.FAILED:
                if getattr(t, "failed_step", None) is not None:
                    terminal_steps.append(int(t.failed_step))
                else:
                    terminal_steps.append(int(max(int(t.deadline_step), 0)))
        delivered_task_last_step = int(max(delivered_steps)) if delivered_steps else -1
        delivered_task_last_time_seconds = float(delivered_task_last_step * self._dt_seconds) if delivered_task_last_step >= 0 else float("nan")
        terminal_task_last_step = int(max(terminal_steps)) if terminal_steps else -1
        terminal_task_last_time_seconds = float(terminal_task_last_step * self._dt_seconds) if terminal_task_last_step >= 0 else float("nan")
        task_end_step = int(terminal_task_last_step)
        task_end_time_seconds = float(terminal_task_last_time_seconds)
        road_observation_event_count_step = int(self._shared_discovery_uav_step + self._shared_discovery_truck_step)
        road_observation_event_count_total = int(self._shared_discovery_uav_total + self._shared_discovery_truck_total)
        semantic_metrics = self._compute_task_semantic_metrics()
        route_plan_v2_audit = getattr(self, "_planner_route_plan_v2", {})
        if not isinstance(route_plan_v2_audit, dict):
            route_plan_v2_audit = {}
        route_plan_v2_contracts = route_plan_v2_audit.get("contracts", {})
        if not isinstance(route_plan_v2_contracts, dict):
            route_plan_v2_contracts = {}
        route_plan_v2_stay_reasons = getattr(
            self, "_planner_route_plan_stay_reason_by_agent", {}
        )
        if not isinstance(route_plan_v2_stay_reasons, dict):
            route_plan_v2_stay_reasons = {}
        bulk_relay_tasks = [
            task
            for task in self.state.tasks.values()
            if self._task_is_bulk_relay(task)
        ]

        return StepResult(
            state=self.state,
            rewards=rewards,
            terminated=self.state.done,
            truncated=self.state.step_index >= self.cfg.max_steps,
            info={
                "hrl_trigger": self.should_trigger_hrl(),
                "accepted_actions": list(action.keys()),
                **v2_metrics(self),
                "route_plan_v2_enabled": int(
                    bool(route_plan_v2_audit.get("enabled", False))
                ),
                "route_plan_v2_version": int(
                    route_plan_v2_audit.get("version", 0) or 0
                ),
                "route_plan_v2_objective": float(
                    route_plan_v2_audit.get("objective", 0.0) or 0.0
                ),
                "route_plan_v2_contract_count": int(
                    len(route_plan_v2_contracts)
                ),
                "route_plan_v2_suffix_repair_count": int(
                    route_plan_v2_audit.get("suffix_repair_count", 0) or 0
                ),
                "route_plan_v2_suffix_repair_success_count": int(
                    route_plan_v2_audit.get(
                        "suffix_repair_success_count", 0
                    )
                    or 0
                ),
                "route_plan_v2_anchor_backup_switch_count": int(
                    route_plan_v2_audit.get(
                        "anchor_backup_switch_count", 0
                    )
                    or 0
                ),
                "route_plan_v2_contract_transfer_count": int(
                    route_plan_v2_audit.get("contract_transfer_count", 0) or 0
                ),
                "route_plan_v2_stalled_contract_transfer_candidate_count": int(
                    route_plan_v2_audit.get(
                        "stalled_contract_transfer_candidate_count", 0
                    )
                    or 0
                ),
                "route_plan_v2_stalled_contract_transfer_replan_count": int(
                    route_plan_v2_audit.get(
                        "stalled_contract_transfer_replan_count", 0
                    )
                    or 0
                ),
                "route_plan_v2_onsite_takeover_count": int(
                    route_plan_v2_audit.get("onsite_takeover_count", 0) or 0
                ),
                "route_plan_v2_routine_opportunity_candidate_count": int(
                    route_plan_v2_audit.get("routine_opportunity_candidate_count", 0)
                    or 0
                ),
                "route_plan_v2_routine_opportunity_transfer_count": int(
                    route_plan_v2_audit.get("routine_opportunity_transfer_count", 0)
                    or 0
                ),
                "route_plan_v2_routine_opportunity_blocked_assist_count": int(
                    route_plan_v2_audit.get(
                        "routine_opportunity_blocked_assist_count", 0
                    )
                    or 0
                ),
                "route_plan_v2_routine_opportunity_blocked_eta_count": int(
                    route_plan_v2_audit.get("routine_opportunity_blocked_eta_count", 0)
                    or 0
                ),
                "route_plan_v2_onsite_capture_count": int(
                    getattr(self, "route_plan_v2_onsite_capture_count", 0)
                ),
                "route_plan_v2_onsite_capture_contract_transfer_count": int(
                    getattr(self, "route_plan_v2_onsite_capture_contract_transfer_count", 0)
                ),
                "route_plan_v2_onsite_capture_preempted_assist_count": int(
                    getattr(self, "route_plan_v2_onsite_capture_preempted_assist_count", 0)
                ),
                "route_plan_v2_deadline_rescue_promotion_count": int(
                    route_plan_v2_audit.get("deadline_rescue_promotion_count", 0) or 0
                ),
                "route_plan_v2_emergency_starvation_promotion_count": int(
                    route_plan_v2_audit.get("emergency_starvation_promotion_count", 0)
                    or 0
                ),
                "route_plan_v2_emergency_launch_watchdog_ready_count": int(
                    route_plan_v2_audit.get("emergency_launch_watchdog_ready_count", 0)
                    or 0
                ),
                "route_plan_v2_emergency_launch_watchdog_force_count": int(
                    route_plan_v2_audit.get("emergency_launch_watchdog_force_count", 0)
                    or 0
                ),
                "route_plan_v2_queue_rescue_assignment_count": int(
                    route_plan_v2_audit.get("queue_rescue_assignment_count", 0)
                    or 0
                ),
                "route_plan_v2_queue_rescue_delivery_count": int(
                    route_plan_v2_audit.get("queue_rescue_delivery_count", 0)
                    or 0
                ),
                "route_plan_v2_direct_safe_secondary_emergency_candidate_count": int(
                    route_plan_v2_audit.get(
                        "direct_safe_secondary_emergency_candidate_count", 0
                    )
                    or 0
                ),
                "route_plan_v2_direct_safe_secondary_emergency_assignment_count": int(
                    route_plan_v2_audit.get(
                        "direct_safe_secondary_emergency_assignment_count", 0
                    )
                    or 0
                ),
                "uav_authoritative_sortie_goal_override_count": int(
                    getattr(
                        self,
                        "uav_authoritative_sortie_goal_override_count",
                        0,
                    )
                ),
                "uav_terminal_delivery_commitment_count": int(
                    getattr(self, "uav_terminal_delivery_commitment_count", 0)
                ),
                "route_plan_v2_lifecycle_turnaround_cost_evaluation_count": int(
                    route_plan_v2_audit.get(
                        "lifecycle_turnaround_cost_evaluation_count", 0
                    )
                    or 0
                ),
                "route_plan_v2_lifecycle_turnaround_cost_total": float(
                    route_plan_v2_audit.get(
                        "lifecycle_turnaround_cost_total", 0.0
                    )
                    or 0.0
                ),
                "route_plan_v2_lexicographic_comparison_count": int(
                    route_plan_v2_audit.get("lexicographic_comparison_count", 0)
                    or 0
                ),
                "route_plan_v2_lexicographic_primary_rejection_count": int(
                    route_plan_v2_audit.get(
                        "lexicographic_primary_rejection_count", 0
                    )
                    or 0
                ),
                "route_plan_v2_disconnect_profile_evaluation_count": int(
                    route_plan_v2_audit.get(
                        "disconnect_profile_evaluation_count", 0
                    )
                    or 0
                ),
                "route_plan_v2_disconnect_protected_task_count": int(
                    route_plan_v2_audit.get("disconnect_protected_task_count", 0)
                    or 0
                ),
                "route_plan_v2_disconnect_predicted_miss_count": int(
                    route_plan_v2_audit.get("disconnect_predicted_miss_count", 0)
                    or 0
                ),
                "route_plan_v2_emergency_balance_trigger_count": int(
                    route_plan_v2_audit.get("emergency_balance_trigger_count", 0)
                    or 0
                ),
                "route_plan_v2_emergency_balance_baseline_max_count": int(
                    route_plan_v2_audit.get(
                        "emergency_balance_baseline_max_count", 0
                    )
                    or 0
                ),
                "route_plan_v2_emergency_capacity_repair_count": int(
                    route_plan_v2_audit.get("emergency_capacity_repair_count", 0)
                    or 0
                ),
                "route_plan_v2_emergency_capacity_contract_move_count": int(
                    route_plan_v2_audit.get(
                        "emergency_capacity_contract_move_count", 0
                    )
                    or 0
                ),
                "route_plan_v2_residual_emergency_handoff_count": int(
                    route_plan_v2_audit.get("residual_emergency_handoff_count", 0)
                    or 0
                ),
                "route_plan_v2_routine_inventory_rebalance_count": int(
                    route_plan_v2_audit.get("routine_inventory_rebalance_count", 0)
                    or 0
                ),
                "route_plan_v2_normal_cleanup_replan_count": int(
                    route_plan_v2_audit.get("normal_cleanup_replan_count", 0)
                    or 0
                ),
                "route_plan_v2_b_orphaned_routine_rescue_count": int(
                    route_plan_v2_audit.get(
                        "b_orphaned_routine_rescue_count", 0
                    )
                    or 0
                ),
                "route_plan_v2_queue_starvation_repair_count": int(
                    route_plan_v2_audit.get("queue_starvation_repair_count", 0)
                    or 0
                ),
                "route_plan_v2_initial_lifeline_ordering_enabled": int(
                    bool(
                        route_plan_v2_audit.get(
                            "initial_lifeline_ordering_enabled", False
                        )
                    )
                ),
                "route_plan_v2_contract_consistency_block_count": int(
                    route_plan_v2_audit.get("contract_consistency_block_count", 0) or 0
                ),
                "routine_multiround_commitment_count": int(
                    self.routine_multiround_commitment_count
                ),
                "routine_multiround_support_block_count": int(
                    self.routine_multiround_support_block_count
                ),
                "route_plan_v2_bulk_relay_task_count": int(
                    len(bulk_relay_tasks)
                ),
                "route_plan_v2_bulk_relay_delivered_count": int(
                    sum(
                        task.status == TaskStatus.DELIVERED
                        for task in bulk_relay_tasks
                    )
                ),
                "route_plan_v2_stay_reasons": dict(
                    route_plan_v2_stay_reasons
                ),
                "invalid_action_count": int(invalid_action_count),
                "invalid_action_mean": float(
                    invalid_action_count / max(len(self.state.agents), 1)
                ),
                "planner_candidate_invalid_count": int(self.planner_candidate_invalid_count_total),
                "pre_dispatch_rejected_count": int(self.pre_dispatch_rejected_count_total),
                "pre_dispatch_repair_success_count": int(self.pre_dispatch_repair_success_count_total),
                "safe_noop_fallback_count": int(self.safe_noop_fallback_count_total),
                "environment_invalid_action_count": int(invalid_action_count),
                "crashed_agents": crashed_agents,
                "reassigned_count": int(reassigned),
                "service_started_count": int(len(started_services)),
                "service_started_by_uav_step": int(service_started_by_uav_step),
                "service_started_by_truck_step": int(service_started_by_truck_step),
                "servicing_agent_count": int(len(servicing_now)),
                "delivered_normal": int(delivered_normal),
                "delivered_emergency": int(delivered_emergency),
                "delivered_by_uav_step": int(delivered_by_uav_step),
                "delivered_emergency_by_uav_step": int(delivered_emergency_by_uav_step),
                "delivered_by_truck_step": int(delivered_by_truck_step),
                "uav_delivered_tasks_total": int(self.uav_delivered_tasks_total),
                "uav_delivered_emergency_total": int(self.uav_delivered_emergency_total),
                "uav_delivery_count_total": int(self.uav_delivery_count_total),
                "truck_delivered_tasks_total": int(self.truck_delivered_tasks_total),
                "truck_normal_supply_units_total": int(truck_normal_supply_units_total),
                "truck_emergency_supply_units_total": int(truck_emergency_supply_units_total),
                "truck_replenish_count_step": int(step_truck_replenish_events),
                "truck_replenish_count_total": int(self.truck_replenish_count_total),
                "truck_empty_trip_count_step": int(step_truck_empty_trip_count),
                "truck_empty_trip_count_total": int(self.truck_empty_trip_count_total),
                "uav_reload_count_step": int(step_uav_reload_events),
                "uav_reload_count_total": int(self.uav_reload_count_total),
                "uav_reload_wait_steps_total": int(self.uav_reload_wait_steps_total),
                "uav_recharge_count_step": int(step_uav_recharge_events),
                "uav_recharge_count_total": int(self.uav_recharge_count_total),
                "uav_empty_flight_count_step": int(step_uav_empty_flight_count),
                "uav_empty_flight_count_total": int(self.uav_empty_flight_count_total),
                "normal_tasks_blocked_by_supply_step": int(step_normal_blocked_supply),
                "normal_tasks_blocked_by_supply_count": int(self.normal_tasks_blocked_by_supply_count),
                "emergency_tasks_blocked_by_supply_step": int(step_emergency_blocked_supply),
                "emergency_tasks_blocked_by_supply_count": int(self.emergency_tasks_blocked_by_supply_count),
                "failed_normal": int(failed_normal),
                "failed_emergency": int(failed_emergency),
                "task_completion_rate": float(
                    sum(
                        1 for t in self.state.tasks.values() if t.status == TaskStatus.DELIVERED
                    )
                    / max(len(self.state.tasks), 1)
                ),
                "completion_rate": float(semantic_metrics.get("overall_completion_rate", 0.0)),
                "overall_completion_rate": float(semantic_metrics.get("overall_completion_rate", 0.0)),
                "routine_bulk_completion_rate": float(semantic_metrics.get("routine_bulk_completion_rate", 0.0)),
                "time_critical_lightweight_completion_rate": float(semantic_metrics.get("time_critical_lightweight_completion_rate", 0.0)),
                "time_critical_on_time_completion_rate": float(semantic_metrics.get("time_critical_on_time_completion_rate", 0.0)),
                "time_critical_on_time_completed_count_total": int(semantic_metrics.get("time_critical_on_time_completed_count_total", 0)),
                "time_critical_completion_time_mean_seconds": float(semantic_metrics.get("time_critical_completion_time_mean_seconds", 0.0)),
                "routine_bulk_completion_time_mean_seconds": float(semantic_metrics.get("routine_bulk_completion_time_mean_seconds", 0.0)),
                "overall_completion_time_mean_seconds": float(semantic_metrics.get("overall_completion_time_mean_seconds", 0.0)),
                "bulk_fulfilled_mass_ratio": float(semantic_metrics.get("bulk_fulfilled_mass_ratio", 0.0)),
                "failed_task_count": int(semantic_metrics.get("failed_task_count", 0)),
                "failed_due_to_lifeline_zero_count": int(semantic_metrics.get("failed_due_to_lifeline_zero_count", 0)),
                "mean_remaining_lifeline_at_service": float(semantic_metrics.get("mean_remaining_lifeline_at_service", 0.0)),
                "mean_remaining_lifeline_bulk": float(semantic_metrics.get("mean_remaining_lifeline_bulk", 0.0)),
                "mean_remaining_lifeline_time_critical": float(semantic_metrics.get("mean_remaining_lifeline_time_critical", 0.0)),
                "mean_remaining_lifeline_at_completion_time_critical": float(semantic_metrics.get("mean_remaining_lifeline_at_completion_time_critical", 0.0)),
                "average_service_delay": float(semantic_metrics.get("average_service_delay", 0.0)),
                "average_service_delay_bulk": float(semantic_metrics.get("average_service_delay_bulk", 0.0)),
                "average_service_delay_time_critical": float(semantic_metrics.get("average_service_delay_time_critical", 0.0)),
                "weighted_service_score": float(semantic_metrics.get("weighted_service_score", 0.0)),
                "routine_bulk_completed_count_total": int(semantic_metrics.get("routine_bulk_completed_count_total", 0)),
                "time_critical_lightweight_completed_count_total": int(semantic_metrics.get("time_critical_lightweight_completed_count_total", 0)),
                "routine_bulk_failed_count_total": int(semantic_metrics.get("routine_bulk_failed_count_total", 0)),
                "time_critical_lightweight_failed_count_total": int(semantic_metrics.get("time_critical_lightweight_failed_count_total", 0)),
                "uav_follow_bind_count_step": int(step_follow_bind_count),
                "uav_follow_steps_step": int(step_follow_steps),
                "uav_charge_energy_gain_step": float(step_follow_charge_energy),
                "uav_low_battery_events_step": int(step_low_battery_events),
                "uav_low_battery_return_success_step": int(step_low_battery_return_success),
                "uav_energy_used_step": float(step_uav_energy_used),
                "uav_energy_used_total": float(self.uav_energy_used_total),
                "crash_count_step": int(step_crash_count),
                "crash_count_total": int(self.crash_count_total),
                "crash_rate_total": float(
                    self.crash_count_total / max(self._num_uavs, 1)
                ),
                "battery_depletion_count_step": int(step_battery_depletion_count),
                "battery_depletion_count_total": int(self.battery_depletion_count_total),
                "uav_survival_rate": float(uav_survival_rate),
                "uav_crash_free_rate": float(uav_crash_free_rate),
                "uav_battery_depletion_free_rate": float(uav_battery_depletion_free_rate),
                "uav_survival_rate_raw": float(uav_survival_rate_raw),
                "uav_crash_free_rate_raw": float(uav_crash_free_rate_raw),
                "uav_terminal_battery_rescue_count_total": int(self.uav_terminal_battery_rescue_count_total),
                "uav_follow_bind_count_total": int(self.follow_bind_count_total),
                "uav_follow_steps_total": int(self.follow_steps_total),
                "uav_charge_energy_gain_total": float(self.follow_charge_energy_total),
                "uav_safe_launch_count_step": int(step_uav_safe_launch),
                "uav_safe_launch_count_total": int(self.uav_safe_launch_count_total),
                "uav_launch_count_step": int(step_uav_launch),
                "uav_launch_count_total": int(self.uav_launch_count_total),
                "uav_launch_battery_fraction_mean": float(uav_launch_battery_fraction_mean),
                "uav_launch_battery_fraction_min": float(uav_launch_battery_fraction_min),
                "uav_unsafe_launch_attempt_count_step": int(step_uav_unsafe_launch_attempt),
                "uav_unsafe_launch_attempt_count_total": int(self.uav_unsafe_launch_attempt_count_total),
                "uav_unsafe_launch_block_count_step": int(step_uav_unsafe_launch_block),
                "uav_unsafe_launch_block_count_total": int(self.uav_unsafe_launch_block_count_total),
                "uav_low_battery_illegal_launch_count_step": int(step_uav_low_battery_illegal_launch),
                "uav_low_battery_illegal_launch_count_total": int(self.uav_low_battery_illegal_launch_count_total),
                "uav_forced_recovery_count_step": int(step_uav_forced_recovery),
                "uav_forced_recovery_count_total": int(self.uav_forced_recovery_count_total),
                "uav_forced_recovery_due_to_low_battery_count_step": int(step_uav_forced_recovery_low_batt),
                "uav_forced_recovery_due_to_low_battery_count_total": int(self.uav_forced_recovery_due_to_low_battery_count_total),
                "uav_task_reject_below_launch_min_count_step": int(step_uav_task_reject_below_launch_min),
                "uav_task_reject_below_launch_min_count": int(self.uav_task_reject_below_launch_min_count),
                "uav_task_reject_not_loaded_count_step": int(step_uav_task_reject_not_loaded),
                "uav_task_reject_not_loaded_count": int(self.uav_task_reject_not_loaded_count),
                "uav_task_reject_recovery_margin_count_step": int(step_uav_task_reject_recovery_margin),
                "uav_task_reject_recovery_margin_count": int(self.uav_task_reject_recovery_margin_count),
                "uav_task_reject_horizon_count_step": int(step_uav_task_reject_horizon),
                "uav_task_reject_horizon_count": int(self.uav_task_reject_horizon_count),
                "uav_task_reject_comm_block_count_step": int(step_uav_task_reject_comm_block),
                "uav_task_reject_comm_block_count": int(self.uav_task_reject_comm_block_count),
                "uav_task_reject_corridor_count_step": int(step_uav_task_reject_corridor),
                "uav_task_reject_corridor_count": int(self.uav_task_reject_corridor_count),
                "uav_launch_direct_safe_count_step": int(step_uav_launch_direct_safe),
                "uav_launch_direct_safe_count": int(self.uav_launch_direct_safe_count),
                "uav_launch_rendezvous_safe_count_step": int(step_uav_launch_rendezvous_safe),
                "uav_launch_rendezvous_safe_count": int(self.uav_launch_rendezvous_safe_count),
                "uav_launch_rendezvous_safe_relaxed_count_step": int(step_uav_launch_rendezvous_safe_relaxed),
                "uav_launch_rendezvous_safe_relaxed_count": int(self.uav_launch_rendezvous_safe_relaxed_count),
                "uav_launch_block_unsafe_count_step": int(step_uav_launch_block_unsafe),
                "uav_launch_block_unsafe_count": int(self.uav_launch_block_unsafe_count),
                "uav_launch_gate_enter_count_step": int(step_uav_launch_gate_enter),
                "uav_launch_gate_enter_count": int(self.uav_launch_gate_enter_count),
                "uav_launch_feasibility_eval_count": int(self.uav_launch_gate_enter_count),
                "relaxed_sortie_selected_count_total": int(self.relaxed_sortie_selected_count_total),
                "relaxed_delivery_completed_count_total": int(self.relaxed_delivery_completed_count_total),
                "relaxed_conversion_rate": float(
                    float(self.relaxed_delivery_completed_count_total)
                    / max(float(self.relaxed_sortie_selected_count_total), 1.0)
                ),
                "uav_launch_gate_direct_safe_count_step": int(step_uav_launch_gate_direct),
                "uav_launch_gate_direct_safe_count": int(self.uav_launch_gate_direct_safe_count),
                "uav_launch_gate_rendezvous_safe_count_step": int(step_uav_launch_gate_rendezvous),
                "uav_launch_gate_rendezvous_safe_count": int(self.uav_launch_gate_rendezvous_safe_count),
                "uav_launch_gate_rendezvous_safe_relaxed_count_step": int(step_uav_launch_gate_rendezvous_relaxed),
                "uav_launch_gate_rendezvous_safe_relaxed_count": int(self.uav_launch_gate_rendezvous_safe_relaxed_count),
                "uav_launch_gate_block_below_launch_min_count_step": int(step_uav_launch_gate_block_below),
                "uav_launch_gate_block_below_launch_min_count": int(self.uav_launch_gate_block_below_launch_min_count),
                "uav_launch_gate_block_recovery_margin_count_step": int(step_uav_launch_gate_block_margin),
                "uav_launch_gate_block_recovery_margin_count": int(self.uav_launch_gate_block_recovery_margin_count),
                "uav_launch_gate_block_corridor_count_step": int(step_uav_launch_gate_block_corridor),
                "uav_launch_gate_block_corridor_count": int(self.uav_launch_gate_block_corridor_count),
                "uav_launch_gate_block_other_count_step": int(step_uav_launch_gate_block_other),
                "uav_launch_gate_block_other_count": int(self.uav_launch_gate_block_other_count),
                "uav_rendezvous_success_count_step": int(step_uav_rendezvous_success),
                "uav_rendezvous_success_count_total": int(self.uav_rendezvous_success_count_total),
                "uav_rendezvous_fail_count_step": int(step_uav_rendezvous_fail),
                "uav_rendezvous_fail_count_total": int(self.uav_rendezvous_fail_count_total),
                "truck_recovery_support_count_step": int(step_truck_recovery_support),
                "truck_recovery_support_count_total": int(self.truck_recovery_support_count_total),
                "truck_emergency_blocked_by_normal_guard_count_step": int(step_truck_emergency_blocked_by_normal_guard),
                "truck_emergency_blocked_by_normal_guard_count": int(self.truck_emergency_blocked_by_normal_guard_count),
                "truck_emergency_relief_override_count_step": int(step_truck_emergency_relief_override),
                "truck_emergency_relief_override_count": int(self.truck_emergency_relief_override_count),
                "truck_emergency_serviceable_count_step": int(step_truck_emergency_serviceable),
                "truck_emergency_serviceable_count": int(self.truck_emergency_serviceable_count),
                "truck_emergency_not_serviceable_count_step": int(step_truck_emergency_not_serviceable),
                "truck_emergency_not_serviceable_count": int(self.truck_emergency_not_serviceable_count),
                "truck_forward_support_count_step": int(step_truck_forward_support),
                "truck_forward_support_count_total": int(self.truck_forward_support_count_total),
                "truck_forward_support_distance_step": float(step_truck_forward_support_distance),
                "truck_forward_support_distance_total": float(self.truck_forward_support_distance_total),
                "unauthorized_support_attempt_count": int(self.unauthorized_support_attempt_count),
                "unauthorized_support_blocked_count": int(self.unauthorized_support_blocked_count),
                "unauthorized_recovery_attempt_count": int(self.unauthorized_recovery_attempt_count),
                "unauthorized_recovery_blocked_count": int(self.unauthorized_recovery_blocked_count),
                "command_rejected_count": int(self.command_rejected_count),
                "command_rejected_reason_launch_unauthorized_count": int(self.command_rejected_reason_launch_unauthorized_count),
                "support_command_count": int(getattr(self, "support_command_count", 0)),
                "support_command_to_launch_count": int(getattr(self, "support_command_to_launch_count", 0)),
                "support_command_to_delivery_count": int(getattr(self, "support_command_to_delivery_count", 0)),
                "safety_recovery_command_count": int(getattr(self, "safety_recovery_command_count", 0)),
                "routine_near_completion_protected_count": int(self.routine_near_completion_protected_count),
                "routine_near_completion_support_blocked_count": int(self.routine_near_completion_support_blocked_count),
                "routine_near_completion_recovery_blocked_count": int(self.routine_near_completion_recovery_blocked_count),
                "routine_near_completion_broken_by_hard_safety_count": int(self.routine_near_completion_broken_by_hard_safety_count),
                "routine_near_completion_broken_by_tc_override_count": int(self.routine_near_completion_broken_by_tc_override_count),
                "routine_near_completion_tc_override_to_launch_count": int(self.routine_near_completion_tc_override_to_launch_count),
                "routine_near_completion_tc_override_to_delivery_count": int(self.routine_near_completion_tc_override_to_delivery_count),
                "routine_near_completion_blocked_tc_support_count": int(self.routine_near_completion_blocked_tc_support_count),
                "routine_near_completion_followed_by_service_start_count": int(self.routine_near_completion_followed_by_service_start_count),
                "routine_near_completion_followed_by_completion_count": int(self.routine_near_completion_followed_by_completion_count),
                "routine_near_completion_tc_override_reject_delay_count": int(self.routine_near_completion_tc_override_reject_delay_count),
                "routine_near_completion_tc_override_reject_no_loaded_uav_count": int(self.routine_near_completion_tc_override_reject_no_loaded_uav_count),
                "routine_near_completion_tc_override_reject_no_candidate_count": int(self.routine_near_completion_tc_override_reject_no_candidate_count),
                "routine_near_completion_tc_override_reject_not_near_launchable_count": int(self.routine_near_completion_tc_override_reject_not_near_launchable_count),
                "routine_near_completion_tc_override_reject_recovery_count": int(self.routine_near_completion_tc_override_reject_recovery_count),
                "routine_near_completion_broken_by_delivery_feasible_tc_override_count": int(self.routine_near_completion_broken_by_delivery_feasible_tc_override_count),
                "tc_override_candidate_count": int(self.tc_override_candidate_count),
                "tc_override_blocked_not_full_sortie_feasible_count": int(self.tc_override_blocked_not_full_sortie_feasible_count),
                "tc_override_blocked_low_recovery_margin_count": int(self.tc_override_blocked_low_recovery_margin_count),
                "tc_override_blocked_low_battery_margin_count": int(self.tc_override_blocked_low_battery_margin_count),
                "tc_override_blocked_recent_reject_count": int(self.tc_override_blocked_recent_reject_count),
                "tc_override_blocked_lifeline_risk_count": int(self.tc_override_blocked_lifeline_risk_count),
                "tc_override_blocked_routine_delay_count": int(self.tc_override_blocked_routine_delay_count),
                "tc_override_to_launch_count": int(self.tc_override_to_launch_count),
                "tc_override_to_delivery_count": int(self.tc_override_to_delivery_count),
                "tc_override_to_forced_recovery_count": int(self.tc_override_to_forced_recovery_count),
                "tc_override_feasibility_mismatch_count": int(self.tc_override_feasibility_mismatch_count),
                "tc_override_predicted_launchable_count": int(self.tc_override_predicted_launchable_count),
                "tc_override_actual_launch_count": int(self.tc_override_actual_launch_count),
                "tc_override_predicted_delivery_feasible_count": int(self.tc_override_predicted_delivery_feasible_count),
                "tc_override_actual_delivery_count": int(self.tc_override_actual_delivery_count),
                "island_task_count_total": int(island_task_count_total),
                "island_task_active_count_step": int(len(step_island_task_ids)),
                "island_task_candidate_count_step": int(step_island_task_candidate),
                "island_task_candidate_count": int(self.island_task_candidate_count),
                "island_task_serviceable_count_step": int(step_island_task_serviceable),
                "island_task_serviceable_count": int(self.island_task_serviceable_count),
                "island_task_launch_block_count_step": int(step_island_task_launch_block),
                "island_task_launch_block_count": int(self.island_task_launch_block_count),
                "island_task_completed_count_step": int(step_island_task_completed),
                "island_task_completed_count_total": int(self.island_task_completed_count_total),
                "uav_island_delivery_count_step": int(step_uav_island_delivery),
                "uav_island_delivery_count_total": int(self.uav_island_delivery_count_total),
                "uav_island_recovery_success_count_step": int(step_uav_island_recovery_success),
                "uav_island_recovery_success_count_total": int(self.uav_island_recovery_success_count_total),
                "uav_docked_retarget_count_step": int(self.uav_docked_retarget_count_step),
                "uav_docked_retarget_count_total": int(self.uav_docked_retarget_count_total),
                "uav_urgent_watchdog_assign_count_step": int(self.uav_urgent_watchdog_assign_count_step),
                "uav_urgent_watchdog_assign_count_total": int(self.uav_urgent_watchdog_assign_count_total),
                "uav_low_battery_events_total": int(self.low_battery_events_total),
                "uav_low_battery_return_success_total": int(
                    self.low_battery_return_success_total
                ),
                "forced_rth_count_step": int(step_forced_rth_count),
                "forced_rth_count_total": int(self.forced_rth_count_total),
                "forced_sortie_limit_return_count": int(step_sortie_limit_hits),
                "forced_sortie_limit_return_total": int(self.sortie_limit_hit_total),
                "sortie_limit_violation_count": int(step_sortie_limit_hits),
                "sortie_limit_violation_total": int(self.sortie_limit_hit_total),
                "uav_forced_takeoff_full_step": int(step_forced_takeoff_full),
                "uav_low_battery_return_success_rate": float(
                    self.low_battery_return_success_total
                    / max(self.low_battery_events_total, 1)
                ),
                "invalid_action_count_total": int(self.invalid_action_count_total),
                "planner_candidate_invalid_count_total": int(self.planner_candidate_invalid_count_total),
                "pre_dispatch_rejected_count_total": int(self.pre_dispatch_rejected_count_total),
                "pre_dispatch_repair_success_count_total": int(self.pre_dispatch_repair_success_count_total),
                "safe_noop_fallback_count_total": int(self.safe_noop_fallback_count_total),
                "environment_invalid_action_count_total": int(self.environment_invalid_action_count_total),
                "invalid_action_count_uav_step": int(invalid_action_count_uav),
                "invalid_action_count_truck_step": int(invalid_action_count_truck),
                "invalid_action_count_uav_total": int(self.invalid_action_count_uav_total),
                "invalid_action_count_truck_total": int(self.invalid_action_count_truck_total),
                "queue_wait_steps_step": int(step_queue_wait_steps),
                "queue_wait_steps_total": int(self.queue_wait_steps_total),
                "queue_time_seconds_step": float(step_queue_wait_steps * self._dt_seconds),
                "queue_time_seconds_total": float(self.queue_wait_steps_total * self._dt_seconds),
                "pbrs_target_switch_count_step": int(step_pbrs_switch_count),
                "pbrs_target_switch_count_total": int(self._pbrs_switch_total),
                "comm_blocked_count": int(sum(1 for v in self.comm_blocked.values() if v)),
                "comm_blackout_count": int(sum(1 for v in self.comm_blocked.values() if v)),
                "comm_blackout_ratio": float(sum(1 for v in self.comm_blocked.values() if v) / max(len(self.comm_blocked), 1)),
                "comm_blackout_agent_observation_count_total": int(self.comm_blackout_agent_observation_count_total),
                "comm_blackout_agent_blocked_count_total": int(self.comm_blackout_agent_blocked_count_total),
                "comm_blackout_agent_time_exposure_ratio": float(
                    self.comm_blackout_agent_blocked_count_total
                    / max(self.comm_blackout_agent_observation_count_total, 1)
                ),
                "comm_blackout_physical_zone_count_total": int(self.comm_blackout_physical_zone_count_total),
                "comm_blackout_goal_zone_count_total": int(self.comm_blackout_goal_zone_count_total),
                "comm_blackout_active_zone_count": int(self._comm_blackout_active_zone_count),
                "comm_blackout_zone_count": int(len(self._comm_blackout_zones)),
                "comm_blackout_nominal_emergency_coverage": float(
                    getattr(self.cfg, "comm_blackout_emergency_coverage", 0.0)
                ),
                "comm_blackout_zone_radius_map_fraction": float(
                    getattr(self.cfg, "comm_blackout_zone_radius_map_fraction", 0.0)
                ),
                "comm_blackout_zone_radius_mean_m": float(
                    np.mean(
                        [float(zone.get("radius_m", 0.0)) for zone in self._comm_blackout_zones]
                    )
                    if self._comm_blackout_zones
                    else 0.0
                ),
                "comm_blackout_start_step": int(getattr(self.cfg, "comm_blackout_start_step", 0)),
                "comm_blackout_duration_steps": int(getattr(self.cfg, "comm_blackout_duration_steps", 0)),
                "comm_blackout_recovery_steps": int(getattr(self.cfg, "comm_blackout_recovery_steps", 0)),
                "comm_blackout_cycle_steps": int(
                    max(getattr(self.cfg, "comm_blackout_duration_steps", 0), 0)
                    + max(getattr(self.cfg, "comm_blackout_recovery_steps", 0), 0)
                ),
                "comm_blackout_duty_cycle": float(
                    max(getattr(self.cfg, "comm_blackout_duration_steps", 0), 0)
                    / max(
                        max(getattr(self.cfg, "comm_blackout_duration_steps", 0), 0)
                        + max(getattr(self.cfg, "comm_blackout_recovery_steps", 0), 0),
                        1,
                    )
                ),
                "comm_blackout_covered_task_count": int(len(self._comm_blackout_zone_task_ids)),
                "comm_blackout_covered_task_ratio": float(
                    len(self._comm_blackout_zone_task_ids)
                    / max(int(self._comm_blackout_emergency_task_count), 1)
                ),
                "comm_blackout_covered_node_count": int(len(self._comm_blackout_zone_node_ids)),
                "comm_blackout_covered_node_ratio": float(
                    len(self._comm_blackout_zone_node_ids)
                    / max(int(self._comm_blackout_emergency_node_count), 1)
                ),
                "comm_blackout_zone_digest": str(self._comm_blackout_zone_digest),
                "blocked_edge_count": int(getattr(self.hazards, "blocked_edge_count_total", getattr(self.hazards, "blocked_edge_count", len(self.topology.blocked_edges)))),
                "blocked_edge_count_stochastic_step": int(getattr(self.hazards, "blocked_edge_count_stochastic_step", 0)),
                "blocked_edge_count_forced_island_step": int(getattr(self.hazards, "blocked_edge_count_forced_island_step", 0)),
                "blocked_edge_count_total_step": int(getattr(self.hazards, "blocked_edge_count_total_step", len(self.topology.blocked_edges))),
                "blocked_edge_count_stochastic": int(getattr(self.hazards, "blocked_edge_count_stochastic", 0)),
                "blocked_edge_count_forced_island": int(getattr(self.hazards, "blocked_edge_count_forced_island", 0)),
                "blocked_edge_count_total": int(getattr(self.hazards, "blocked_edge_count_total", len(self.topology.blocked_edges))),
                "blocked_ratio_stochastic_step": float(getattr(self.hazards, "blocked_ratio_stochastic", 0.0)),
                "blocked_ratio_forced_island_step": float(getattr(self.hazards, "blocked_ratio_forced_island", 0.0)),
                "blocked_ratio_total_step": float(getattr(self.hazards, "blocked_ratio_total", blocked_ratio)),
                "blocked_ratio_stochastic_final": float(getattr(self.hazards, "blocked_ratio_stochastic", 0.0)),
                "blocked_ratio_forced_island_final": float(getattr(self.hazards, "blocked_ratio_forced_island", 0.0)),
                "blocked_ratio_total_final": float(getattr(self.hazards, "blocked_ratio_total", blocked_ratio)),
                "blockage_target_ratio_step": float(getattr(self.hazards, "blockage_target_ratio_step", 0.0)),
                "blockage_target_ratio_stochastic_step": float(getattr(self.hazards, "blockage_target_ratio_stochastic_step", 0.0)),
                "blockage_current_ratio_stochastic_step": float(getattr(self.hazards, "blockage_current_ratio_stochastic_step", 0.0)),
                "blockage_gap_step": float(getattr(self.hazards, "blockage_gap_step", 0.0)),
                "blockage_global_gate_step": float(getattr(self.hazards, "blockage_global_gate_step", 0.0)),
                "blockage_target_ratio_final": float(getattr(self.hazards, "blockage_target_ratio_step", 0.0)),
                "blockage_target_ratio_stochastic_final": float(getattr(self.hazards, "blockage_target_ratio_stochastic_step", 0.0)),
                "blockage_curve_B_inf": float(getattr(self.hazards, "blockage_curve_B_inf", 0.0)),
                "blockage_curve_tau_steps": float(getattr(self.hazards, "blockage_curve_tau_steps", 0.0)),
                "newly_blocked_edge_count_step": int(getattr(self.hazards, "newly_blocked_edge_count_step", 0)),
                "newly_blocked_edge_count_total": int(getattr(self.hazards, "newly_blocked_edge_count_total", 0)),
                "shared_road_awareness_mode": str(getattr(self.cfg, "road_awareness_mode", "perfect")),
                "shared_road_awareness_enabled": bool(getattr(self.cfg, "road_shared_awareness_enabled", True)),
                "shared_blocked_edge_count": int(len(self._shared_known_blocked_edges)),
                "shared_map_update_event_step": bool(self._shared_map_update_event_step),
                "shared_map_update_count_total": int(self._shared_map_update_count_total),
                "shared_map_new_blocked_step": int(self._shared_map_new_blocked_step),
                "shared_map_new_blocked_total": int(self._shared_map_new_blocked_total),
                "shared_map_cleared_step": int(self._shared_map_cleared_step),
                "shared_map_cleared_total": int(self._shared_map_cleared_total),
                "shared_discovery_uav_step": int(self._shared_discovery_uav_step),
                "shared_discovery_uav_total": int(self._shared_discovery_uav_total),
                "shared_discovery_truck_step": int(self._shared_discovery_truck_step),
                "shared_discovery_truck_total": int(self._shared_discovery_truck_total),
                "road_observation_event_count_step": int(road_observation_event_count_step),
                "road_observation_event_count_total": int(road_observation_event_count_total),
                "road_observation_event_uav_count_total": int(self._shared_discovery_uav_total),
                "road_observation_event_truck_count_total": int(self._shared_discovery_truck_total),
                "planner_refresh_map_update_step": bool(self.planner_refresh_map_update_step),
                "planner_replan_due_to_new_road_info_count_total": int(self.planner_replan_due_to_new_road_info_count_total),
                "planner_last_replan_reason": str(self.planner_last_replan_reason),
                "unknown_blocked_edge_hit_step": int(self._unknown_blocked_edge_hit_step),
                "unknown_blocked_edge_hit_total": int(self._unknown_blocked_edge_hit_total),
                "shared_map_last_update_reason": str(self._shared_last_update_reason),
                "truck_inventory_kg_current_mean": float(truck_inventory_kg_current_mean),
                "uav_loaded_fraction": float(uav_loaded_fraction),
                "truck_replenish_event_flag": bool(step_truck_replenish_events > 0),
                "uav_reload_event_flag": bool(step_uav_reload_events > 0),
                "rainfall_mean": rain_mean,
                "rain_severity_p95_mmh": float(rain_severity_p95_mmh),
                "wind_mean": wind_mean,
                "wind_severity_p95_mps": float(wind_severity_p95_mps),
                "blocked_ratio": blocked_ratio,
                "epicenter_node": epicenter,
                "risk_spike": bool(risk_spike),
                "tasks_terminal": bool(tasks_terminal),
                "uav_settled_for_termination": bool(uav_settled),
                "dt_seconds": float(self._dt_seconds),
                "avg_degree": float(self.topology.average_degree()),
                "percolation_phase": str(self.hazards.last_percolation_phase),
                "percolation_lambda": float(self.hazards.last_lambda),
                "macro_block_prob_mean": float(self.hazards.last_pmacro_mean),
                "step_block_prob_mean": float(self.hazards.last_pstep_mean),
                "block_factor_bldg_mean": float(self.hazards.last_bldg_mean),
                "block_factor_infra_mean": float(self.hazards.last_infra_mean),
                "block_factor_length_norm_mean": float(self.hazards.last_length_norm_mean),
                "block_factor_length_ref_m": float(self.hazards.edge_len_ref_m),
                "reward_step_total": float(reward_step_total),
                "reward_invalid_total": float(reward_invalid_total),
                "reward_idle_total": float(reward_idle_total),
                "reward_delivery_total": float(reward_delivery_total),
                "reward_timeout_total": float(reward_timeout_total),
                "reward_pbrs_total": float(reward_pbrs_total),
                "reward_crash_total": float(reward_crash_total),
                "reward_discover_total": float(reward_discover_total),
                "reward_docking_total": float(reward_docking_total),
                "reward_pickup_total": float(reward_pickup_total),
                "reward_delivery_shared_total": float(reward_delivery_shared_total),
                "reward_uav_emergency_bonus_total": float(
                    reward_uav_emergency_bonus_total
                ),
                "uav_headwind_mean_step": float(mean_headwind),
                "uav_rain_mean_step": float(mean_rain),
                "uav_discovered_blocked_step": int(step_uav_discovered_blocked),
                "uav_discovered_blocked_total": int(self._uav_discovered_blocked_total),
                "wind_failure_event_count": int(step_wind_failure_event_count),
                "wind_failure_event_count_total": int(self.wind_failure_event_total),
                "wind_failure_risk_accum": float(step_wind_failure_risk_accum),
                "wind_failure_risk_accum_total": float(self.wind_failure_risk_accum_total),
                "uav_following_count": int(uav_following_count),
                "uav_follow_with_goal_count": int(uav_follow_with_goal_count),
                "uav_follow_near_goal_count": int(uav_follow_near_goal_count),
                "uav_follow_far_goal_count": int(uav_follow_far_goal_count),
                "truck_distance_total_m": float(truck_distance_total_m),
                "uav_distance_total_m": float(uav_distance_total_m),
                "fleet_distance_total_m": float(fleet_distance_total_m),
                "delivered_task_last_step": int(delivered_task_last_step),
                "delivered_task_last_time_seconds": float(delivered_task_last_time_seconds),
                "terminal_task_last_step": int(terminal_task_last_step),
                "terminal_task_last_time_seconds": float(terminal_task_last_time_seconds),
                "task_end_step": int(task_end_step),
                "task_end_time_seconds": float(task_end_time_seconds),
                "uav_goal_task_count_step": int(step_uav_goal_task_count),
                "triggered_replans_step": int(self.triggered_replans_step),
                "triggered_replans_total": int(self.triggered_replans_total),
                "assignment_assigned_total": int(self.last_assignment_summary.get("assigned_total", 0)),
                "assignment_assigned_truck": int(self.last_assignment_summary.get("assigned_truck", 0)),
                "assignment_assigned_uav": int(self.last_assignment_summary.get("assigned_uav", 0)),
                "uav_terminal_zone_count_step": int(step_uav_terminal_zone_count),
                "uav_delivery_zone_count_step": int(step_uav_delivery_zone_count),
            },
        )

    def observe(self) -> Dict[str, List[float]]:
        # Step-1 observation shape baseline, will be replaced by full design.
        obs: Dict[str, List[float]] = {}
        pending_normal = sum(
            1
            for t in self.state.tasks.values()
            if t.status == TaskStatus.PENDING
            and t.kind == TaskKind.NORMAL
        )
        pending_emergency = sum(
            1
            for t in self.state.tasks.values()
            if t.status == TaskStatus.PENDING
            and t.kind == TaskKind.EMERGENCY
        )
        total_tasks = max(1, len(self.state.tasks))
        for aid, s in self.state.agents.items():
            node = int(s.node or 0)
            weather = self._agent_weather_sample(aid)
            assigned = self._pbrs_target_task(aid)
            blocked = bool(self.comm_blocked.get(aid, False))
            goal_dx = 0.0
            goal_dy = 0.0
            if assigned is not None:
                goal_dx, goal_dy, _ = self._agent_task_rel(aid, assigned)
            global_blocked_ratio = 0.0 if blocked else float(self._decision_blocked_ratio())
            global_pending_normal = 0.0 if blocked else float(pending_normal / total_tasks)
            global_pending_emergency = 0.0 if blocked else float(pending_emergency / total_tasks)
            if s.kind == AgentKind.TRUCK:
                total_nb = len(self.topology.adjacency.get(node, set()))
                valid_nb = len(self._decision_neighbors(node))
                nb_ratio = float(valid_nb / max(total_nb, 1))
            else:
                nb_ratio = self._uav_visible_edge_ratio(
                    aid, radius_m=float(self.cfg.uav_monitor_radius_m)
                )
            obs[aid] = [
                float(self.state.step_index / max(self.cfg.max_steps, 1)),
                float(s.battery),
                float(s.crashed),
                float(1.0 if s.kind == AgentKind.UAV else 0.0),
                float(1.0 if s.kind == AgentKind.TRUCK else 0.0),
                float(node / max(self._num_nodes - 1, 1)),
                float(weather.rain / 30.0),
                float(weather.wind / 20.0),
                float(weather.quake),
                float(goal_dx),
                float(goal_dy),
                global_blocked_ratio,
                global_pending_normal,
                global_pending_emergency,
                self._agent_task_distance_norm(aid),
                float(1.0 if (assigned is not None and assigned.kind == TaskKind.EMERGENCY) else 0.0),
                float(1.0 if s.follow_target is not None else 0.0),
                float(1.0 if blocked else 0.0),
                float(1.0 if self.state.hazard.risk_spike else 0.0),
                nb_ratio,
            ]
        return obs

    def observe_task_matrix(self) -> Dict[str, List[List[float]]]:
        """
        Returns fixed-size task matrix per agent:
        [task_attention_slots, task_feat_dim],
        features=[norm_dx, norm_dy, norm_dist, emergency_flag, is_recommended].
        """
        per_agent: Dict[str, List[List[float]]] = {}
        for aid in self.state.agents:
            s = self.state.agents[aid]
            blocked = bool(self.comm_blocked.get(aid, False))
            rec_tid = self._effective_goals.get(
                str(aid), self._recommended_goals.get(str(aid), None)
            )
            mat: List[List[float]] = []

            # Virtual task: nearest truck (for UAV energy-aware learning, no hard-coded battery rule).
            if s.kind == AgentKind.UAV:
                ax, ay = self._agent_xy(aid)
                nearest_tid: Optional[str] = None
                nearest_dist = float("inf")
                nearest_dx = 0.0
                nearest_dy = 0.0
                for tid, ts in self.state.agents.items():
                    if ts.kind != AgentKind.TRUCK:
                        continue
                    tx, ty = self._agent_xy(tid)
                    dx = float(tx - ax)
                    dy = float(ty - ay)
                    d = float(np.hypot(dx, dy))
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest_tid = str(tid)
                        nearest_dx = dx
                        nearest_dy = dy
                if nearest_tid is not None:
                    mat.append(
                        [
                            float(np.clip(nearest_dx / 3000.0, -1.0, 1.0)),
                            float(np.clip(nearest_dy / 3000.0, -1.0, 1.0)),
                            float(np.clip(nearest_dist / 3000.0, 0.0, 1.0)),
                            0.0,  # emergency_flag
                            0.0,  # is_recommended
                        ]
                    )

            if not blocked:
                active_tasks = [
                    t for t in self.state.tasks.values() if self._task_visible_to_agent(aid, t)
                ]
                # Emergency first then earliest deadline.
                active_tasks.sort(
                    key=lambda t: (
                        0 if t.kind == TaskKind.EMERGENCY else 1,
                        t.deadline_step,
                    )
                )
                # Localized by ranking nearest tasks from active pool.
                ranked = sorted(
                    active_tasks,
                    key=lambda t: self._agent_distance_to_task(aid, t),
                )
                max_real_rows = max(0, int(self.task_attention_slots) - len(mat))
                for t in ranked[:max_real_rows]:
                    dx, dy, d = self._agent_task_rel(aid, t)
                    is_rec = 1.0 if (rec_tid is not None and str(t.task_id) == str(rec_tid)) else 0.0
                    mat.append(
                        [
                            float(dx),
                            float(dy),
                            float(d),
                            float(1.0 if t.kind == TaskKind.EMERGENCY else 0.0),
                            float(is_rec),
                        ]
                    )
            while len(mat) < self.task_attention_slots:
                mat.append([0.0, 0.0, 1.0, 0.0, 0.0])
            per_agent[aid] = mat
        return per_agent

    def observe_task_slots(self) -> Dict[str, List[Optional[str]]]:
        """
        Returns task ids aligned with observe_task_matrix rows for each agent.
        Row i in task matrix corresponds to slots[aid][i].
        """
        per_agent: Dict[str, List[Optional[str]]] = {}
        for aid in self.state.agents:
            s = self.state.agents[aid]
            blocked = bool(self.comm_blocked.get(aid, False))
            slots: List[Optional[str]] = []

            # Keep strict alignment with observe_task_matrix(): virtual nearest-truck slot first for UAV.
            if s.kind == AgentKind.UAV:
                ax, ay = self._agent_xy(aid)
                nearest_tid: Optional[str] = None
                nearest_dist = float("inf")
                for tid, ts in self.state.agents.items():
                    if ts.kind != AgentKind.TRUCK:
                        continue
                    tx, ty = self._agent_xy(tid)
                    d = float(np.hypot(tx - ax, ty - ay))
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest_tid = str(tid)
                if nearest_tid is not None:
                    slots.append(nearest_tid)

            if not blocked:
                active_tasks = [
                    t for t in self.state.tasks.values() if self._task_visible_to_agent(aid, t)
                ]
                active_tasks.sort(
                    key=lambda t: (
                        0 if t.kind == TaskKind.EMERGENCY else 1,
                        t.deadline_step,
                    )
                )
                ranked = sorted(
                    active_tasks,
                    key=lambda t: self._agent_distance_to_task(aid, t),
                )
                max_real_slots = max(0, int(self.task_attention_slots) - len(slots))
                for t in ranked[:max_real_slots]:
                    slots.append(str(t.task_id))
            while len(slots) < self.task_attention_slots:
                slots.append(None)
            per_agent[aid] = slots
        return per_agent

    def legal_actions(self) -> Dict[str, object]:
        legal: Dict[str, object] = {}
        for aid, s in self.state.agents.items():
            if s.kind == AgentKind.TRUCK:
                node = int(s.node or 0)
                self._sync_truck_inventory_fields(s)
                neighbors = list(self._decision_neighbors(node))
                mask = [1 if i in neighbors else 0 for i in range(self._num_nodes)]
                requires_depot = bool(self._truck_requires_depot(aid))
                replenish_in_progress = bool(int(getattr(s, "truck_replenish_timer", 0)) > 0)
                if replenish_in_progress:
                    neighbors = []
                    mask = [0 for _ in range(self._num_nodes)]
                elif requires_depot and int(node) != 0 and neighbors:
                    best_nb = min(
                        neighbors,
                        key=lambda nb: self._decision_shortest_path_distance(
                            int(nb), 0
                        ),
                    )
                    neighbors = [int(best_nb)]
                    mask = [1 if i == int(best_nb) else 0 for i in range(self._num_nodes)]
                legal[aid] = {
                    "type": "discrete_node",
                    "stay": True,
                    "neighbors": neighbors,
                    "mask": mask,
                    "requires_depot_replenish": bool(requires_depot),
                    "replenish_in_progress": bool(replenish_in_progress),
                }
            else:
                self._sync_uav_payload_fields(s)
                can_takeoff_safe = True
                launch_reason = "n/a"
                if s.follow_target is not None:
                    can_takeoff_safe, launch_reason, _ = self._uav_launch_gate_check(str(aid))
                legal[aid] = {
                    "type": "continuous_xy",
                    "bind_or_takeoff": True,
                    "vmax": self.cfg.uav_max_speed_mps,
                    "can_direct_emergency": bool(self._uav_loaded(aid)),
                    "uav_needs_reload": bool(getattr(s, "uav_needs_reload_flag", False)),
                    "can_takeoff_safe": bool(can_takeoff_safe),
                    "launch_gate_reason": str(launch_reason),
                    "forced_recovery": bool(self._uav_forced_rth_latch.get(str(aid), False)),
                }
        return legal

    def should_trigger_hrl(self) -> bool:
        by_interval = self.state.step_index % max(self.cfg.hrl_interval, 1) == 0
        by_risk = bool(self.state.hazard.risk_spike)
        by_shared_map = bool(
            getattr(self.cfg, "road_shared_replan_on_update", True)
            and bool(getattr(self, "_shared_map_update_event_step", False))
        )
        return bool(by_interval or by_risk or by_shared_map)
























































