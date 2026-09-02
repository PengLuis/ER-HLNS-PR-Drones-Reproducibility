from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class MapComplexitySpec:
    name: str
    map_size_m: float
    node_count: int
    min_node_spacing_m: float
    redundant_edge_radius_m: float
    redundant_edge_prob: float
    target_avg_degree: float = 3.5
    max_degree: int = 5
    generator_mode: str = "legacy"
    quality_gate_max_attempts: int = 1
    target_stats: Optional[Dict[str, Tuple[float, float]]] = None


MAP_COMPLEXITY_PRESETS: Dict[str, MapComplexitySpec] = {
    "M": MapComplexitySpec(
        name="M",
        map_size_m=5000.0,
        node_count=40,
        min_node_spacing_m=300.0,
        redundant_edge_radius_m=850.0,
        redundant_edge_prob=0.55,
        target_avg_degree=3.5,
        max_degree=5,
        generator_mode="legacy",
        quality_gate_max_attempts=1,
        target_stats=None,
    ),
    "L": MapComplexitySpec(
        name="L",
        map_size_m=15000.0,
        node_count=400,
        min_node_spacing_m=170.0,
        redundant_edge_radius_m=1800.0,
        redundant_edge_prob=0.90,
        target_avg_degree=2.92,
        max_degree=5,
        generator_mode="mesoscopic",
        quality_gate_max_attempts=8,
        target_stats={
            "num_nodes": (320.0, 380.0),
            "num_edges": (460.0, 540.0),
            "avg_degree": (2.8, 3.3),
            "median_edge_length_m": (450.0, 800.0),
            "p90_edge_length_m": (1800.0, 3200.0),
            "leaf_fraction": (0.0, 0.08),
            "deg3_fraction": (0.55, 0.70),
            "deg4_fraction": (0.18, 0.30),
            "deg_gt4_fraction": (0.0, 0.08),
            "arterial_length_share": (0.10, 0.15),
            "collector_length_share": (0.25, 0.35),
            "local_length_share": (0.50, 0.60),
            "crossing_fraction": (0.0, 0.05),
            "off_axis_edge_fraction": (0.20, 0.35),
            "builtup_area_fraction": (0.40, 0.55),
            "barrier_area_fraction": (0.08, 0.18),
        },
    ),
}


class DisasterMapGraph:
    """
    Synthetic disaster road graph used as a shared map for truck/UAV.

    M-complexity keeps the lightweight random geometric backbone.
    L-complexity is generated as a mesoscopic decision graph: built-up driven,
    hierarchical roads (arterial/collector/local), near-planar geometry,
    and regenerate-on-fail quality gating.
    """

    def __init__(
        self,
        seed: int = 0,
        map_size_m: float = 5000.0,
        node_count: int = 40,
        min_node_spacing_m: float = 300.0,
        redundant_edge_radius_m: float = 1000.0,
        redundant_edge_prob: float = 0.8,
        target_avg_degree: float = 3.5,
        max_degree: int = 5,
        map_complexity: str = "",
        quality_gate_max_attempts: int = 1,
        target_stats: Optional[Dict[str, Tuple[float, float]]] = None,
        l_map_variant: str = "L_v1_baseline",
        l_map_acceptance_mode: str = "realism_first",
        l_min_node_spacing_m: float = 220.0,
        l_min_gateway_spacing_m: float = 300.0,
        l_min_arterial_junction_spacing_m: float = 350.0,
        l_collinear_triangle_angle_deg: float = 165.0,
        task_normal_count: int = 8,
        task_emergency_count: int = 12,
        task_min_spacing_m: float = 240.0,
        use_map_cache: bool = False,
        map_cache_dir: str = "data/maps_core/L_map",
        map_cache_size: int = 20,
        map_cache_index: Optional[int] = None,
    ) -> None:
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.map_size_m = float(map_size_m)
        self.node_count = int(node_count)
        self.min_node_spacing_m = float(min_node_spacing_m)
        self.redundant_edge_radius_m = float(redundant_edge_radius_m)
        self.redundant_edge_prob = float(np.clip(redundant_edge_prob, 0.0, 1.0))
        self.target_avg_degree = float(max(float(target_avg_degree), 1.0))
        self.max_degree = int(max(int(max_degree), 2))
        self.map_complexity = str(map_complexity).upper().strip()
        self.quality_gate_max_attempts = int(max(int(quality_gate_max_attempts), 1))
        self.target_stats = dict(target_stats or {})
        self.l_map_variant = str(l_map_variant).strip() or "L_v1_baseline"
        self.l_map_acceptance_mode = str(l_map_acceptance_mode).strip().lower() or "realism_first"
        self.l_min_node_spacing_m = float(max(float(l_min_node_spacing_m), 80.0))
        self.l_min_gateway_spacing_m = float(max(float(l_min_gateway_spacing_m), self.l_min_node_spacing_m))
        self.l_min_arterial_junction_spacing_m = float(max(float(l_min_arterial_junction_spacing_m), self.l_min_gateway_spacing_m))
        self.l_collinear_triangle_angle_deg = float(np.clip(float(l_collinear_triangle_angle_deg), 150.0, 179.5))
        self.task_normal_count = int(max(int(task_normal_count), 0))
        self.task_emergency_count = int(max(int(task_emergency_count), 0))
        self.task_min_spacing_m = float(max(float(task_min_spacing_m), 40.0))
        self.use_map_cache = bool(use_map_cache)
        self.map_cache_dir = str(map_cache_dir).strip() or "data/maps_core/L_map"
        self.map_cache_size = int(max(int(map_cache_size), 1))
        self.map_cache_index = None if map_cache_index is None else int(max(int(map_cache_index), 0))
        self.scene_payload: Dict[str, Any] = {}
        self.realism_metrics: Dict[str, float] = {}
        self.cache_schema_version: str = "realism_first_v8_routine_access"
        if self._should_use_mesoscopic_l() and not self.target_stats:
            preset = MAP_COMPLEXITY_PRESETS.get("L")
            if preset is not None and preset.target_stats is not None:
                self.target_stats = dict(preset.target_stats)
            if self.quality_gate_max_attempts <= 1 and preset is not None:
                self.quality_gate_max_attempts = int(max(int(preset.quality_gate_max_attempts), 1))

        self.node_xy: Dict[int, Tuple[float, float]] = {}
        self.node_attrs: Dict[int, Dict[str, Any]] = {}
        self.graph: nx.Graph = nx.Graph()
        self.euclidean_dist_matrix: Optional[np.ndarray] = None
        self.shortest_path_matrix: Optional[np.ndarray] = None
        self.map_stats: Dict[str, float] = {}
        self.map_stats_quality_passed: bool = True
        self.map_stats_attempts: int = 1

        if self._should_use_mesoscopic_l():
            loaded = False
            if self.use_map_cache:
                loaded = bool(self._try_load_cached_map())
            if not loaded:
                self._generate_mesoscopic_l()
                if self.use_map_cache:
                    self._save_cached_map()
        else:
            self._generate_nodes()
            self._generate_edges()
            self.map_stats = self._compute_map_stats()
        self._cache_distances()

    @classmethod
    def from_complexity(cls, complexity: str, seed: int = 0) -> "DisasterMapGraph":
        key = str(complexity).upper().strip()
        if key not in MAP_COMPLEXITY_PRESETS:
            raise ValueError(f"unknown complexity: {complexity}; choices={list(MAP_COMPLEXITY_PRESETS.keys())}")
        spec = MAP_COMPLEXITY_PRESETS[key]
        return cls(
            seed=seed,
            map_size_m=spec.map_size_m,
            node_count=spec.node_count,
            min_node_spacing_m=spec.min_node_spacing_m,
            redundant_edge_radius_m=spec.redundant_edge_radius_m,
            redundant_edge_prob=spec.redundant_edge_prob,
            target_avg_degree=spec.target_avg_degree,
            max_degree=spec.max_degree,
            map_complexity=spec.name,
            quality_gate_max_attempts=spec.quality_gate_max_attempts,
            target_stats=spec.target_stats,
        )

    def _should_use_mesoscopic_l(self) -> bool:
        if self.map_complexity == "L":
            return True
        return bool(self.map_size_m >= 12000.0 and self.node_count >= 220)

    def _generate_nodes(self) -> None:
        if self.node_count < 1:
            raise ValueError("node_count must be >= 1")
        center = 0.5 * self.map_size_m
        self.node_xy[0] = (center, center)
        self.node_attrs[0] = {"node_role": "depot_hub", "builtup_intensity": 0.5, "barrier_proximity": 0.0}

        max_trials = 200_000
        trials = 0
        next_id = 1
        while next_id < self.node_count:
            trials += 1
            if trials > max_trials:
                raise RuntimeError(
                    f"node sampling failed after {max_trials} trials; "
                    f"increase map_size_m or decrease min_node_spacing_m/node_count"
                )
            x = float(self.rng.uniform(0.0, self.map_size_m))
            y = float(self.rng.uniform(0.0, self.map_size_m))

            ok = True
            for px, py in self.node_xy.values():
                if float(np.hypot(x - px, y - py)) < self.min_node_spacing_m:
                    ok = False
                    break
            if not ok:
                continue
            self.node_xy[next_id] = (x, y)
            self.node_attrs[next_id] = {"node_role": "ordinary", "builtup_intensity": float(np.clip(self.rng.uniform(0.1, 0.6), 0.0, 1.0)), "barrier_proximity": float(np.clip(self.rng.uniform(0.0, 0.2), 0.0, 1.0))}
            next_id += 1

        self.graph.clear()
        for i in range(self.node_count):
            attrs = self.node_attrs.get(i, {})
            self.graph.add_node(i, x=float(self.node_xy[i][0]), y=float(self.node_xy[i][1]), **attrs)

    def _generate_edges(self) -> None:
        n = len(self.node_xy)
        if n <= 1:
            return

        xy = np.array([self.node_xy[i] for i in range(n)], dtype=np.float64)
        diff = xy[:, None, :] - xy[None, :, :]
        dist = np.linalg.norm(diff, axis=2)

        # Build a local-road candidate graph first (kNN + radius neighborhood),
        # then take MST on that local graph. This avoids unrealistically frequent
        # long direct A-B links and encourages multi-hop connectivity.
        local_k = int(np.clip(int(round(self.target_avg_degree)) + 1, 3, max(3, self.max_degree + 1)))
        local_radius = float(max(self.redundant_edge_radius_m, 2.2 * self.min_node_spacing_m))
        candidates_graph = nx.Graph()
        candidates_graph.add_nodes_from(range(n))
        for i in range(n):
            order = np.argsort(dist[i])
            for j in order[1 : local_k + 1]:
                candidates_graph.add_edge(int(i), int(j), weight=float(dist[i, int(j)]))
            for j in range(i + 1, n):
                dij = float(dist[i, j])
                if dij <= local_radius:
                    candidates_graph.add_edge(int(i), int(j), weight=dij)

        # Keep candidate graph connected with nearest inter-component bridges.
        comps = [set(c) for c in nx.connected_components(candidates_graph)]
        while len(comps) > 1:
            best_pair: Optional[Tuple[int, int]] = None
            best_d = float("inf")
            for idx in range(len(comps) - 1):
                c1 = comps[idx]
                for jdx in range(idx + 1, len(comps)):
                    c2 = comps[jdx]
                    for a in c1:
                        for b in c2:
                            dij = float(dist[int(a), int(b)])
                            if dij < best_d:
                                best_d = dij
                                best_pair = (int(a), int(b))
            if best_pair is None:
                break
            a, b = best_pair
            candidates_graph.add_edge(a, b, weight=float(dist[a, b]))
            comps = [set(c) for c in nx.connected_components(candidates_graph)]

        mst = nx.minimum_spanning_tree(candidates_graph, algorithm="kruskal", weight="weight")
        for u, v, data in mst.edges(data=True):
            self.graph.add_edge(int(u), int(v), weight=float(data["weight"]))

        # Controlled redundant links to avoid local near-clique structures.
        target_edges = int(
            np.clip(
                int(round(0.5 * self.target_avg_degree * float(n))),
                max(n - 1, 1),
                max(int(0.5 * self.max_degree * float(n)), n - 1),
            )
        )
        degrees = {int(k): int(v) for k, v in self.graph.degree()}

        candidates_local: List[Tuple[float, int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                if self.graph.has_edge(i, j):
                    continue
                dij = float(dist[i, j])
                if dij <= float(self.redundant_edge_radius_m):
                    jitter = float(self.rng.uniform(0.0, 1e-6))
                    candidates_local.append((dij + jitter, int(i), int(j)))
        candidates_local.sort(key=lambda x: float(x[0]))

        for _, i, j in candidates_local:
            if self.graph.number_of_edges() >= target_edges:
                break
            if degrees.get(i, 0) >= self.max_degree or degrees.get(j, 0) >= self.max_degree:
                continue
            if float(self.rng.uniform(0.0, 1.0)) > float(self.redundant_edge_prob):
                continue
            self.graph.add_edge(int(i), int(j), weight=float(dist[i, j]))
            degrees[i] = int(degrees.get(i, 0) + 1)
            degrees[j] = int(degrees.get(j, 0) + 1)

        # Rare long direct links are allowed as highway-like shortcuts,
        # but with very small probability and small budget.
        if self.graph.number_of_edges() < target_edges:
            expressway_prob = float(np.clip(0.08 * self.redundant_edge_prob, 0.0, 0.08))
            expressway_budget = int(max(1, round(0.04 * float(target_edges))))
            expressway_min_len = float(max(1.35 * local_radius, 0.28 * self.map_size_m))
            expressway_count = 0
            candidates_express: List[Tuple[float, int, int]] = []
            for i in range(n):
                for j in range(i + 1, n):
                    if self.graph.has_edge(i, j):
                        continue
                    dij = float(dist[i, j])
                    if dij >= expressway_min_len:
                        jitter = float(self.rng.uniform(0.0, 1e-6))
                        candidates_express.append((dij + jitter, int(i), int(j)))
            candidates_express.sort(key=lambda x: float(x[0]))
            for _, i, j in candidates_express:
                if self.graph.number_of_edges() >= target_edges or expressway_count >= expressway_budget:
                    break
                if degrees.get(i, 0) >= self.max_degree or degrees.get(j, 0) >= self.max_degree:
                    continue
                if float(self.rng.uniform(0.0, 1.0)) > expressway_prob:
                    continue
                self.graph.add_edge(int(i), int(j), weight=float(dist[i, j]))
                degrees[i] = int(degrees.get(i, 0) + 1)
                degrees[j] = int(degrees.get(j, 0) + 1)
                expressway_count += 1

        # If too sparse, fill by nearest links under degree cap.
        if self.graph.number_of_edges() < target_edges:
            candidates_all: List[Tuple[float, int, int]] = []
            for i in range(n):
                for j in range(i + 1, n):
                    if self.graph.has_edge(i, j):
                        continue
                    jitter = float(self.rng.uniform(0.0, 1e-6))
                    candidates_all.append((float(dist[i, j]) + jitter, int(i), int(j)))
            candidates_all.sort(key=lambda x: float(x[0]))
            for _, i, j in candidates_all:
                if self.graph.number_of_edges() >= target_edges:
                    break
                if self.graph.has_edge(int(i), int(j)):
                    continue
                if degrees.get(i, 0) >= self.max_degree or degrees.get(j, 0) >= self.max_degree:
                    continue
                self.graph.add_edge(int(i), int(j), weight=float(dist[i, j]))
                degrees[i] = int(degrees.get(i, 0) + 1)
                degrees[j] = int(degrees.get(j, 0) + 1)

    def _generate_mesoscopic_l(self) -> None:
        attempts = int(max(self.quality_gate_max_attempts, 1))
        best_graph: Optional[nx.Graph] = None
        best_xy: Optional[Dict[int, Tuple[float, float]]] = None
        best_attrs: Optional[Dict[int, Dict[str, Any]]] = None
        best_stats: Optional[Dict[str, float]] = None
        best_scene: Optional[Dict[str, Any]] = None
        best_realism: Optional[Dict[str, float]] = None
        best_penalty = float("inf")

        for k in range(attempts):
            rng = np.random.default_rng(self.seed + 1009 * k)
            node_xy, node_attrs, graph, builtup_frac, barrier_frac = self._build_mesoscopic_candidate(rng)

            # Prune near-collinear redundant triangle shortcuts before scoring.
            self._prune_collinear_redundant_edges(node_xy=node_xy, graph=graph)
            # Reinforce cluster access redundancy for realism-first maps.
            self._reinforce_major_cluster_access(node_xy=node_xy, node_attrs=node_attrs, graph=graph)
            # Re-prune after reinforcement to remove newly introduced near-collinear shortcuts.
            self._prune_collinear_redundant_edges(node_xy=node_xy, graph=graph)

            stats = self._compute_map_stats_from(
                node_xy=node_xy,
                graph=graph,
                builtup_area_fraction=builtup_frac,
                barrier_area_fraction=barrier_frac,
            )
            realism = self._compute_realism_bundle(
                node_xy=node_xy,
                node_attrs=node_attrs,
                graph=graph,
                stats=stats,
                rng=rng,
            )

            stats_pen = self._stats_penalty(stats)
            realism_pen = float(realism.get('realism_penalty', 0.0))
            if str(self.l_map_acceptance_mode).strip().lower() == 'realism_first':
                penalty = float(realism_pen + 0.15 * stats_pen)
            else:
                penalty = float(stats_pen)

            if penalty < best_penalty:
                best_penalty = float(penalty)
                best_graph = graph
                best_xy = node_xy
                best_attrs = node_attrs
                best_stats = dict(stats)
                best_scene = dict(realism.get('scene_payload', {}))
                best_realism = dict(realism.get('metrics', {}))

            # realism-first: accept once all hard realism constraints pass.
            if str(self.l_map_acceptance_mode).strip().lower() == 'realism_first':
                if bool(realism.get('realism_first_pass', False)):
                    break
            else:
                if stats_pen <= 0.0:
                    break

        if best_graph is None or best_xy is None or best_attrs is None:
            raise RuntimeError('failed to synthesize mesoscopic L map')

        # Ensure L hard floors by realism-safe densify.
        min_nodes = int(self.target_stats.get("num_nodes", (320.0, 380.0))[0]) if self.target_stats else 320
        min_edges = int(self.target_stats.get("num_edges", (460.0, 540.0))[0]) if self.target_stats else 460
        best_xy, best_attrs, best_graph = self._densify_l_to_minimum_requirements(
            node_xy=best_xy,
            node_attrs=best_attrs,
            graph=best_graph,
            min_nodes=min_nodes,
            min_edges=min_edges,
        )
        best_xy, best_attrs, best_graph = self._reindex_graph(node_xy=best_xy, node_attrs=best_attrs, graph=best_graph)
        self._ensure_l_edge_attrs_complete(graph=best_graph, node_xy=best_xy, node_attrs=best_attrs)
        self._calibrate_l_road_hierarchy(graph=best_graph)

        self.graph = best_graph
        self.node_xy = best_xy
        self.node_attrs = best_attrs
        self.node_count = int(len(best_xy))
        self.map_stats = dict(best_stats or {})
        self.realism_metrics = dict(best_realism or {})
        self.scene_payload = dict(best_scene or {})
        rebuilt_stats = self._compute_map_stats_from(
            node_xy=self.node_xy,
            graph=self.graph,
            builtup_area_fraction=float(self.map_stats.get("builtup_area_fraction", 0.0)),
            barrier_area_fraction=float(self.map_stats.get("barrier_area_fraction", 0.0)),
        )
        self.map_stats = dict(rebuilt_stats)
        self.map_stats.update(self.realism_metrics)
        diag = self._realism_hard_soft_metrics(node_xy=self.node_xy, graph=self.graph)
        self.map_stats.update(diag)
        hard_ok = bool(
            float(diag.get("realized_node_count", 0.0)) >= float(min_nodes)
            and float(diag.get("realized_edge_count", 0.0)) >= float(min_edges)
            and float(diag.get("connected_component_count", 99.0)) == 1.0
            and float(diag.get("depot_degree", 0.0)) >= 3.0
            and float(diag.get("missing_edge_attr_count", 1.0)) == 0.0
        )
        soft_ok = bool(
            float(diag.get("avg_degree", 0.0)) <= 3.5
            and float(diag.get("deg_gt4_fraction", 1.0)) <= 0.10
            and float(diag.get("crossing_fraction", 1.0)) <= 0.05
        )
        self.map_stats["map_stats_hard_gate_passed"] = 1.0 if hard_ok else 0.0
        self.map_stats["map_stats_soft_gate_passed"] = 1.0 if soft_ok else 0.0
        self.map_stats_quality_passed = bool(hard_ok and soft_ok)
        if not hard_ok:
            raise RuntimeError(
                "L map hard gate failed: "
                f"nodes={diag.get('realized_node_count', 0.0)}, edges={diag.get('realized_edge_count', 0.0)}, "
                f"components={diag.get('connected_component_count', 0.0)}, depot_degree={diag.get('depot_degree', 0.0)}, "
                f"missing_edge_attr_count={diag.get('missing_edge_attr_count', 0.0)}"
            )
        self.map_stats_attempts = int(attempts)


    def _cache_file_path(self) -> Path:
        idx = int(self.map_cache_index) if self.map_cache_index is not None else int(self.seed % max(self.map_cache_size, 1))
        return Path(self.map_cache_dir) / f"L_map_{idx:02d}.json"

    def _try_load_cached_map(self) -> bool:
        if not self._should_use_mesoscopic_l():
            return False
        p = self._cache_file_path()
        if not p.exists():
            return False
        try:
            payload = json.loads(p.read_text(encoding='utf-8'))
            if str(payload.get('cache_schema_version', '')) != str(self.cache_schema_version):
                return False
            node_xy = {int(k): (float(v[0]), float(v[1])) for k, v in dict(payload.get('node_xy', {})).items()}
            node_attrs = {int(k): dict(v) for k, v in dict(payload.get('node_attrs', {})).items()}
            graph = nx.Graph()
            for nid, xy in node_xy.items():
                attrs = dict(node_attrs.get(int(nid), {}))
                graph.add_node(int(nid), x=float(xy[0]), y=float(xy[1]), **attrs)
            for e in payload.get('edges', []):
                u = int(e.get('u')); v = int(e.get('v'))
                if u == v:
                    continue
                graph.add_edge(min(u, v), max(u, v), **dict(e.get('data', {})))
            if graph.number_of_nodes() <= 0:
                return False
            self._ensure_l_edge_attrs_complete(graph=graph, node_xy=node_xy, node_attrs=node_attrs)
            diag = self._realism_hard_soft_metrics(node_xy=node_xy, graph=graph)
            if (
                float(diag.get("realized_node_count", 0.0)) < 320.0
                or float(diag.get("realized_edge_count", 0.0)) < 460.0
                or float(diag.get("connected_component_count", 99.0)) != 1.0
                or float(diag.get("depot_degree", 0.0)) < 3.0
                or float(diag.get("missing_edge_attr_count", 1.0)) > 0.0
            ):
                return False
            self.graph = graph
            self.node_xy = node_xy
            self.node_attrs = node_attrs
            self.node_count = int(len(node_xy))
            self.map_stats = dict(payload.get('map_stats', {}))
            self.scene_payload = dict(payload.get('scene_payload', {}))
            self.realism_metrics = dict(payload.get('realism_metrics', {}))
            self.map_stats.update(diag)
            self.map_stats_quality_passed = bool(payload.get('map_stats_quality_passed', False))
            self.map_stats_attempts = int(payload.get('map_stats_attempts', 1))
            return True
        except Exception:
            return False

    def _save_cached_map(self) -> None:
        if not self._should_use_mesoscopic_l():
            return
        p = self._cache_file_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        edges = []
        for u, v, d in self.graph.edges(data=True):
            edges.append({'u': int(u), 'v': int(v), 'data': dict(d)})
        payload = {
            'seed': int(self.seed),
            'cache_schema_version': str(self.cache_schema_version),
            'variant': str(self.l_map_variant),
            'acceptance_mode': str(self.l_map_acceptance_mode),
            'node_xy': {str(int(k)): [float(v[0]), float(v[1])] for k, v in self.node_xy.items()},
            'node_attrs': {str(int(k)): dict(v) for k, v in self.node_attrs.items()},
            'edges': edges,
            'map_stats': dict(self.map_stats),
            'scene_payload': dict(self.scene_payload),
            'realism_metrics': dict(self.realism_metrics),
            'map_stats_quality_passed': bool(self.map_stats_quality_passed),
            'map_stats_attempts': int(self.map_stats_attempts),
        }
        p.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')

    def _add_edge_with_l_attrs(
        self,
        *,
        graph: nx.Graph,
        node_xy: Dict[int, Tuple[float, float]],
        node_attrs: Dict[int, Dict[str, Any]],
        u: int,
        v: int,
        road_class: str = "local",
    ) -> bool:
        if int(u) == int(v):
            return False
        a, b = (int(u), int(v)) if int(u) < int(v) else (int(v), int(u))
        if graph.has_edge(a, b):
            return False
        ax, ay = node_xy[a]
        bx, by = node_xy[b]
        length = float(np.hypot(ax - bx, ay - by))
        if length <= 1e-6:
            return False
        rc = str(road_class).strip().lower() or "local"
        if rc == "arterial":
            speed, lanes, cap = 58.0, 3, "high"
        elif rc == "collector":
            speed, lanes, cap = 40.0, 2, "medium"
        else:
            rc = "local"
            speed, lanes, cap = 30.0, 1, "low"
        builtup = float(
            np.clip(
                0.5
                * (
                    float(node_attrs.get(a, {}).get("builtup_intensity", 0.0))
                    + float(node_attrs.get(b, {}).get("builtup_intensity", 0.0))
                ),
                0.0,
                1.0,
            )
        )
        barrier = float(
            np.clip(
                0.5
                * (
                    float(node_attrs.get(a, {}).get("barrier_proximity", 0.0))
                    + float(node_attrs.get(b, {}).get("barrier_proximity", 0.0))
                ),
                0.0,
                1.0,
            )
        )
        graph.add_edge(
            a,
            b,
            weight=float(length),
            length_m=float(length),
            road_class=rc,
            travel_speed_kph=float(speed),
            lanes=int(lanes),
            capacity_class=str(cap),
            bridge_or_tunnel=False,
            barrier_exposure=float(barrier),
            builtup_exposure=float(builtup),
            orientation_bin=int(self._orientation_bin(ax, ay, bx, by)),
        )
        return True

    def _edge_missing_l_attrs(self, data: Dict[str, Any]) -> bool:
        required = (
            "weight",
            "length_m",
            "road_class",
            "travel_speed_kph",
            "lanes",
            "capacity_class",
            "bridge_or_tunnel",
            "barrier_exposure",
            "builtup_exposure",
            "orientation_bin",
        )
        for k in required:
            if k not in data:
                return True
        return False

    def _count_missing_edge_attrs(self, graph: nx.Graph) -> int:
        miss = 0
        for _, _, data in graph.edges(data=True):
            if self._edge_missing_l_attrs(dict(data)):
                miss += 1
        return int(miss)

    def _ensure_l_edge_attrs_complete(
        self,
        *,
        graph: nx.Graph,
        node_xy: Dict[int, Tuple[float, float]],
        node_attrs: Dict[int, Dict[str, Any]],
    ) -> None:
        for u, v, data in list(graph.edges(data=True)):
            a, b = int(u), int(v)
            ax, ay = node_xy[a]
            bx, by = node_xy[b]
            length = float(data.get("length_m", data.get("weight", np.hypot(ax - bx, ay - by))))
            data["weight"] = float(length)
            data["length_m"] = float(length)
            rc = str(data.get("road_class", "local")).strip().lower() or "local"
            if rc not in {"local", "collector", "arterial"}:
                rc = "local"
            data["road_class"] = rc
            if "travel_speed_kph" not in data:
                data["travel_speed_kph"] = 58.0 if rc == "arterial" else (40.0 if rc == "collector" else 30.0)
            if "lanes" not in data:
                data["lanes"] = 3 if rc == "arterial" else (2 if rc == "collector" else 1)
            if "capacity_class" not in data:
                data["capacity_class"] = "high" if rc == "arterial" else ("medium" if rc == "collector" else "low")
            data["bridge_or_tunnel"] = bool(data.get("bridge_or_tunnel", False))
            if "barrier_exposure" not in data:
                data["barrier_exposure"] = float(
                    np.clip(
                        0.5
                        * (
                            float(node_attrs.get(a, {}).get("barrier_proximity", 0.0))
                            + float(node_attrs.get(b, {}).get("barrier_proximity", 0.0))
                        ),
                        0.0,
                        1.0,
                    )
                )
            if "builtup_exposure" not in data:
                data["builtup_exposure"] = float(
                    np.clip(
                        0.5
                        * (
                            float(node_attrs.get(a, {}).get("builtup_intensity", 0.0))
                            + float(node_attrs.get(b, {}).get("builtup_intensity", 0.0))
                        ),
                        0.0,
                        1.0,
                    )
                )
            if "orientation_bin" not in data:
                data["orientation_bin"] = int(self._orientation_bin(ax, ay, bx, by))

    def _calibrate_l_road_hierarchy(self, graph: nx.Graph) -> None:
        """Promote long synthetic L links into a realistic road hierarchy.

        The mesoscopic generator creates good geometry but can leave too many
        links labeled as local roads. RC maps preserve OSM road classes, so we
        should preserve a comparable hierarchy for synthetic L instead of
        making every link behave like a local street.
        """
        edges = []
        total_len = 0.0
        for u, v, data in graph.edges(data=True):
            length = float(data.get("length_m", data.get("weight", 0.0)))
            if length <= 0.0:
                continue
            edges.append((length, int(u), int(v), data))
            total_len += float(length)
        if total_len <= 1e-9 or not edges:
            return

        target_art = float(self.target_stats.get("arterial_length_share", (0.10, 0.15))[0]) if self.target_stats else 0.10
        target_col = float(self.target_stats.get("collector_length_share", (0.25, 0.35))[0]) if self.target_stats else 0.25
        target_art = float(np.clip(target_art, 0.04, 0.22))
        target_col = float(np.clip(target_col, 0.12, 0.48))

        def set_class(data: Dict[str, Any], rc: str) -> None:
            data["road_class"] = str(rc)
            data["travel_speed_kph"] = 58.0 if rc == "arterial" else (40.0 if rc == "collector" else 30.0)
            data["lanes"] = 3 if rc == "arterial" else (2 if rc == "collector" else 1)
            data["capacity_class"] = "high" if rc == "arterial" else ("medium" if rc == "collector" else "low")

        # Start from a clean local baseline, then promote the longest links.
        for _length, _u, _v, data in edges:
            set_class(data, "local")

        promoted_art = 0.0
        for length, _u, _v, data in sorted(edges, reverse=True, key=lambda x: x[0]):
            if promoted_art / total_len >= target_art:
                break
            set_class(data, "arterial")
            promoted_art += float(length)

        promoted_col = 0.0
        for length, _u, _v, data in sorted(edges, reverse=True, key=lambda x: x[0]):
            if str(data.get("road_class", "local")) == "arterial":
                continue
            if promoted_col / total_len >= target_col:
                break
            set_class(data, "collector")
            promoted_col += float(length)

    def _realism_hard_soft_metrics(
        self,
        *,
        node_xy: Dict[int, Tuple[float, float]],
        graph: nx.Graph,
    ) -> Dict[str, float]:
        n = int(graph.number_of_nodes())
        m = int(graph.number_of_edges())
        deg = np.array([float(graph.degree(v)) for v in graph.nodes()], dtype=np.float64) if n > 0 else np.array([], dtype=np.float64)
        lengths = []
        for u, v, data in graph.edges(data=True):
            ax, ay = node_xy[int(u)]
            bx, by = node_xy[int(v)]
            lengths.append(float(data.get("length_m", data.get("weight", np.hypot(ax - bx, ay - by)))))
        arr = np.array(lengths, dtype=np.float64) if lengths else np.array([], dtype=np.float64)
        comp_count = int(nx.number_connected_components(graph)) if n > 0 else 0
        depot = None
        for nid in graph.nodes():
            if str(graph.nodes[int(nid)].get("node_role", "ordinary")) == "depot_hub":
                depot = int(nid)
                break
        depot_deg = int(graph.degree(depot)) if depot is not None else 0
        avg_deg = float((2.0 * m / n) if n > 0 else 0.0)
        crossing = float(self._compute_crossing_fraction(node_xy=node_xy, graph=graph))
        deg_gt4 = float(np.mean(deg > 4.0)) if deg.size else 0.0
        return {
            "realized_node_count": float(n),
            "realized_edge_count": float(m),
            "avg_degree": float(avg_deg),
            "depot_degree": float(depot_deg),
            "connected_component_count": float(comp_count),
            "dead_end_count": float(np.sum(deg <= 1.0)) if deg.size else 0.0,
            "bridge_count": float(len(list(nx.bridges(graph)))) if m > 0 else 0.0,
            "missing_edge_attr_count": float(self._count_missing_edge_attrs(graph)),
            "crossing_fraction": float(crossing),
            "deg_gt4_fraction": float(deg_gt4),
            "median_edge_length_m": float(np.median(arr)) if arr.size else 0.0,
            "p90_edge_length_m": float(np.quantile(arr, 0.90)) if arr.size else 0.0,
        }

    def _densify_l_to_minimum_requirements(
        self,
        *,
        node_xy: Dict[int, Tuple[float, float]],
        node_attrs: Dict[int, Dict[str, Any]],
        graph: nx.Graph,
        min_nodes: int,
        min_edges: int,
    ) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, Dict[str, Any]], nx.Graph]:
        # Step-1: node densify by splitting long local/collector edges.
        while int(graph.number_of_nodes()) < int(min_nodes):
            candidates: List[Tuple[float, int, int]] = []
            for u, v, data in graph.edges(data=True):
                rc = str(data.get("road_class", "local"))
                if rc not in {"local", "collector"}:
                    continue
                length = float(data.get("length_m", data.get("weight", 0.0)))
                if length >= 900.0:
                    candidates.append((length, int(u), int(v)))
            if not candidates:
                for u, v, data in graph.edges(data=True):
                    rc = str(data.get("road_class", "local"))
                    if rc not in {"local", "collector"}:
                        continue
                    length = float(data.get("length_m", data.get("weight", 0.0)))
                    if length >= 450.0:
                        candidates.append((length, int(u), int(v)))
            if not candidates:
                break
            candidates.sort(reverse=True, key=lambda x: float(x[0]))
            _, u, v = candidates[0]
            if not graph.has_edge(u, v):
                continue
            old = dict(graph.edges[u, v])
            x = 0.5 * (node_xy[u][0] + node_xy[v][0])
            y = 0.5 * (node_xy[u][1] + node_xy[v][1])
            nid = int(max(node_xy.keys()) + 1)
            node_xy[nid] = (float(x), float(y))
            node_attrs[nid] = {
                "node_role": "ordinary",
                "builtup_intensity": float(
                    np.clip(
                        0.5
                        * (
                            float(node_attrs.get(u, {}).get("builtup_intensity", 0.0))
                            + float(node_attrs.get(v, {}).get("builtup_intensity", 0.0))
                        ),
                        0.0,
                        1.0,
                    )
                ),
                "barrier_proximity": float(
                    np.clip(
                        0.5
                        * (
                            float(node_attrs.get(u, {}).get("barrier_proximity", 0.0))
                            + float(node_attrs.get(v, {}).get("barrier_proximity", 0.0))
                        ),
                        0.0,
                        1.0,
                    )
                ),
            }
            graph.add_node(nid, x=float(x), y=float(y), **node_attrs[nid])
            graph.remove_edge(u, v)
            rc = str(old.get("road_class", "local"))
            self._add_edge_with_l_attrs(graph=graph, node_xy=node_xy, node_attrs=node_attrs, u=u, v=nid, road_class=rc)
            self._add_edge_with_l_attrs(graph=graph, node_xy=node_xy, node_attrs=node_attrs, u=nid, v=v, road_class=rc)
        # Step-2: edge densify with realism constraints.
        max_degree = int(max(4, self.max_degree))
        guard = 0
        while int(graph.number_of_edges()) < int(min_edges):
            guard += 1
            if guard > 20000:
                break
            best: Optional[Tuple[float, int, int]] = None
            nodes = [int(n) for n in graph.nodes()]
            for i in range(len(nodes) - 1):
                a = int(nodes[i])
                for j in range(i + 1, len(nodes)):
                    b = int(nodes[j])
                    if graph.has_edge(a, b):
                        continue
                    if int(graph.degree(a)) >= max_degree or int(graph.degree(b)) >= max_degree:
                        continue
                    d = float(np.hypot(node_xy[a][0] - node_xy[b][0], node_xy[a][1] - node_xy[b][1]))
                    if d < 150.0 or d > 1650.0:
                        continue
                    if self._edge_crosses_existing(node_xy=node_xy, graph=graph, u=a, v=b):
                        continue
                    # avoid collinear shortcut.
                    collinear_shortcut = False
                    for nb in graph.neighbors(a):
                        c = int(nb)
                        if c == b:
                            continue
                        if graph.has_edge(min(c, b), max(c, b)) and self._is_near_collinear_triangle(c, a, b, node_xy):
                            collinear_shortcut = True
                            break
                    if collinear_shortcut:
                        continue
                    score = d + 120.0 * abs(int(graph.degree(a)) - int(graph.degree(b)))
                    if best is None or score < float(best[0]):
                        best = (float(score), a, b)
            if best is None:
                break
            _, a, b = best
            self._add_edge_with_l_attrs(graph=graph, node_xy=node_xy, node_attrs=node_attrs, u=a, v=b, road_class="collector")
            # protect crossing / high-degree realism from runaway densify
            ms = self._realism_hard_soft_metrics(node_xy=node_xy, graph=graph)
            if float(ms.get("crossing_fraction", 0.0)) > 0.08 or float(ms.get("deg_gt4_fraction", 0.0)) > 0.16:
                if graph.has_edge(a, b):
                    graph.remove_edge(a, b)
                break
        self._ensure_l_edge_attrs_complete(graph=graph, node_xy=node_xy, node_attrs=node_attrs)
        return node_xy, node_attrs, graph

    def _is_near_collinear_triangle(self, a: int, b: int, c: int, node_xy: Dict[int, Tuple[float, float]]) -> bool:
        ax, ay = node_xy[int(a)]
        bx, by = node_xy[int(b)]
        cx, cy = node_xy[int(c)]
        v1 = np.array([ax - bx, ay - by], dtype=np.float64)
        v2 = np.array([cx - bx, cy - by], dtype=np.float64)
        n1 = float(np.linalg.norm(v1)); n2 = float(np.linalg.norm(v2))
        if n1 <= 1e-6 or n2 <= 1e-6:
            return False
        cosv = float(np.dot(v1, v2) / max(n1 * n2, 1e-9))
        cosv = float(np.clip(cosv, -1.0, 1.0))
        ang = float(np.degrees(np.arccos(cosv)))
        if ang < float(self.l_collinear_triangle_angle_deg):
            return False
        # b should lie between a and c roughly.
        ac = float(np.hypot(ax - cx, ay - cy))
        ab = float(np.hypot(ax - bx, ay - by))
        bc = float(np.hypot(cx - bx, cy - by))
        return bool(abs((ab + bc) - ac) <= 0.10 * max(ac, 1.0))

    def _prune_collinear_redundant_edges(self, node_xy: Dict[int, Tuple[float, float]], graph: nx.Graph) -> None:
        changed = True
        guard = 0
        while changed and guard < 3:
            guard += 1
            changed = False
            for b in list(graph.nodes()):
                nbs = [int(n) for n in graph.neighbors(int(b))]
                if len(nbs) < 2:
                    continue
                for i in range(len(nbs) - 1):
                    a = int(nbs[i])
                    for j in range(i + 1, len(nbs)):
                        c = int(nbs[j])
                        if not graph.has_edge(min(a, c), max(a, c)):
                            continue
                        if not self._is_near_collinear_triangle(a, int(b), c, node_xy):
                            continue
                        eab = graph.edges[min(a, int(b)), max(a, int(b))]
                        ebc = graph.edges[min(int(b), c), max(int(b), c)]
                        eac = graph.edges[min(a, c), max(a, c)]
                        rc_ab = str(eab.get('road_class', 'collector'))
                        rc_bc = str(ebc.get('road_class', 'collector'))
                        rc_ac = str(eac.get('road_class', 'collector'))
                        if rc_ac not in {rc_ab, rc_bc, 'collector', 'local'}:
                            continue
                        graph.remove_edge(min(a, c), max(a, c))
                        changed = True
                        break
                    if changed:
                        break
                if changed:
                    break

    def _detect_major_clusters(self, node_attrs: Dict[int, Dict[str, Any]], graph: nx.Graph) -> List[Set[int]]:
        bnodes = [int(n) for n in graph.nodes() if float(node_attrs.get(int(n), {}).get('builtup_intensity', 0.0)) >= 0.42]
        if not bnodes:
            return []
        sg = graph.subgraph(bnodes)
        comps = [set(int(x) for x in comp) for comp in nx.connected_components(sg)]
        min_size = int(max(8, round(0.03 * max(graph.number_of_nodes(), 1))))
        return [c for c in comps if len(c) >= min_size]

    def _reinforce_major_cluster_access(self, node_xy: Dict[int, Tuple[float, float]], node_attrs: Dict[int, Dict[str, Any]], graph: nx.Graph) -> None:
        major = self._detect_major_clusters(node_attrs=node_attrs, graph=graph)
        if not major:
            return
        depot = None
        for n in graph.nodes():
            if str(graph.nodes[int(n)].get('node_role', 'ordinary')) == 'depot_hub':
                depot = int(n)
                break
        if depot is None:
            return

        for comp in major:
            gateways = []
            for n in comp:
                for nb in graph.neighbors(int(n)):
                    if int(nb) not in comp:
                        gateways.append(int(n))
                        break
            gateways = sorted(set(gateways))
            if len(gateways) >= 2:
                continue
            # add one extra gateway-edge from cluster to outside backbone node
            cand_inside = sorted(list(comp), key=lambda nid: int(graph.degree(int(nid))), reverse=True)[:20]
            outside = [int(n) for n in graph.nodes() if int(n) not in comp and str(graph.nodes[int(n)].get('node_role', 'ordinary')) in {'arterial_junction','collector_junction','area_gateway','depot_hub','bottleneck'}]
            best = None
            best_d = float('inf')
            for a in cand_inside:
                for b in outside:
                    if graph.has_edge(min(int(a), int(b)), max(int(a), int(b))):
                        continue
                    if self._edge_crosses_existing(node_xy=node_xy, graph=graph, u=int(a), v=int(b)):
                        continue
                    d = float(np.hypot(node_xy[int(a)][0] - node_xy[int(b)][0], node_xy[int(a)][1] - node_xy[int(b)][1]))
                    if d < 450.0 or d > 2600.0:
                        continue
                    if d < best_d:
                        best_d = d
                        best = (int(a), int(b))
            if best is None:
                continue
            a, b = best
            length = float(best_d)
            rc = 'collector' if length <= 1600.0 else 'arterial'
            graph.add_edge(
                min(a, b), max(a, b),
                weight=length,
                length_m=length,
                road_class=rc,
                travel_speed_kph=40.0 if rc == 'collector' else 55.0,
                lanes=2,
                capacity_class='medium' if rc == 'collector' else 'high',
                bridge_or_tunnel=False,
                barrier_exposure=0.0,
                builtup_exposure=float(np.clip(0.5 * (node_attrs[a].get('builtup_intensity', 0.0) + node_attrs[b].get('builtup_intensity', 0.0)), 0.0, 1.0)),
                orientation_bin=int(self._orientation_bin(*node_xy[a], *node_xy[b])),
            )

    @staticmethod
    def _ellipse_intensity(x: float, y: float, zone: Dict[str, float]) -> float:
        cx, cy = float(zone['cx']), float(zone['cy'])
        major, minor = max(float(zone['major']), 1.0), max(float(zone['minor']), 1.0)
        ang = float(zone['ang'])
        ca, sa = float(np.cos(ang)), float(np.sin(ang))
        dx, dy = float(x - cx), float(y - cy)
        u = ca * dx + sa * dy
        v = -sa * dx + ca * dy
        g = float(np.exp(-0.5 * ((u / major) ** 2 + (v / minor) ** 2)))
        return float(np.clip(g, 0.0, 1.0))

    def _generate_hazard_fields(self, node_xy: Dict[int, Tuple[float, float]], node_attrs: Dict[int, Dict[str, Any]], rng: np.random.Generator) -> Dict[str, Dict[str, float]]:
        m = float(self.map_size_m)
        center = (0.5 * m, 0.5 * m)
        # Rain swath: closer to barrier/high barrier exposure proxy.
        bar_nodes = sorted(node_xy.keys(), key=lambda nid: float(node_attrs[int(nid)].get('barrier_proximity', 0.0)), reverse=True)
        r0 = int(bar_nodes[0]) if bar_nodes else 0
        rx, ry = node_xy[int(r0)] if bar_nodes else center
        rain = {
            'cx': float(rx),
            'cy': float(ry),
            'major': float(rng.uniform(0.20 * m, 0.35 * m)),
            'minor': float(rng.uniform(0.08 * m, 0.20 * m)),
            'ang': float(rng.uniform(0.0, np.pi)),
        }
        # Wind swath: elongated and spatially separated from rain core.
        wx = float(np.clip((m - float(rx)) + rng.uniform(-0.08 * m, 0.08 * m), 0.12 * m, 0.88 * m))
        wy = float(np.clip((m - float(ry)) + rng.uniform(-0.08 * m, 0.08 * m), 0.12 * m, 0.88 * m))
        wind = {
            'cx': wx,
            'cy': wy,
            'major': float(rng.uniform(0.28 * m, 0.48 * m)),
            'minor': float(rng.uniform(0.10 * m, 0.22 * m)),
            'ang': float((rain['ang'] + rng.uniform(0.50, 1.25)) % np.pi),
        }
        # Quake field: broad but not all-overlap with rain/wind cores.
        qx = float(np.clip(center[0] + rng.uniform(-0.18 * m, 0.18 * m), 0.16 * m, 0.84 * m))
        qy = float(np.clip(center[1] + rng.uniform(-0.18 * m, 0.18 * m), 0.16 * m, 0.84 * m))
        quake = {
            'cx': qx,
            'cy': qy,
            'major': float(rng.uniform(0.35 * m, 0.55 * m)),
            'minor': float(rng.uniform(0.20 * m, 0.40 * m)),
            'ang': float(rng.uniform(0.0, np.pi)),
        }
        return {'rain': rain, 'wind': wind, 'quake': quake}

    def _count_spacing_violations(self, node_xy: Dict[int, Tuple[float, float]], node_attrs: Dict[int, Dict[str, Any]]) -> Tuple[int, int, int]:
        ids = sorted(int(k) for k in node_xy.keys())
        v_node = 0
        v_gate = 0
        v_art = 0
        for i in range(len(ids) - 1):
            a = ids[i]
            xa, ya = node_xy[a]
            ra = str(node_attrs.get(a, {}).get('node_role', 'ordinary'))
            for j in range(i + 1, len(ids)):
                b = ids[j]
                xb, yb = node_xy[b]
                rb = str(node_attrs.get(b, {}).get('node_role', 'ordinary'))
                d = float(np.hypot(xa - xb, ya - yb))
                if d < float(self.l_min_node_spacing_m):
                    v_node += 1
                if ('gateway' in ra or 'gateway' in rb) and d < float(self.l_min_gateway_spacing_m):
                    v_gate += 1
                if ('arterial_junction' in ra or 'arterial_junction' in rb) and d < float(self.l_min_arterial_junction_spacing_m):
                    v_art += 1
        return int(v_node), int(v_gate), int(v_art)

    def _sample_tasks_realism(self, node_xy: Dict[int, Tuple[float, float]], node_attrs: Dict[int, Dict[str, Any]], graph: nx.Graph, hazards: Dict[str, Dict[str, float]], major_clusters: List[Set[int]], rng: np.random.Generator) -> Dict[str, Any]:
        depot = None
        for nid in graph.nodes():
            if str(graph.nodes[int(nid)].get('node_role', 'ordinary')) == 'depot_hub':
                depot = int(nid)
                break
        nodes = [int(n) for n in graph.nodes() if int(n) != int(depot) and int(graph.degree(int(n))) > 0]
        if not nodes:
            return {'normal': [], 'emergency': [], 'task_cluster_balance_score': 0.0, 'task_hazard_exposure_share': 0.0, 'task_min_spacing_violation_count': 0}

        cluster_of = {}
        for cid, comp in enumerate(major_clusters):
            for n in comp:
                cluster_of[int(n)] = int(cid)

        def haz_score(nid: int) -> float:
            x, y = node_xy[int(nid)]
            r = self._ellipse_intensity(x, y, hazards['rain'])
            w = self._ellipse_intensity(x, y, hazards['wind'])
            q = self._ellipse_intensity(x, y, hazards['quake'])
            return float(np.clip(0.34 * r + 0.33 * w + 0.33 * q, 0.0, 1.0))

        def w_normal(nid: int) -> float:
            b = float(node_attrs[int(nid)].get('builtup_intensity', 0.0))
            g = 1.0 if 'gateway' in str(node_attrs[int(nid)].get('node_role', '')) else 0.0
            h = haz_score(int(nid))
            deg_score = float(np.clip(float(graph.degree(int(nid))) / 5.0, 0.0, 1.0))
            if depot is not None and int(depot) in node_xy:
                x, y = node_xy[int(nid)]
                dx, dy = node_xy[int(depot)]
                depot_dist = float(np.hypot(float(x) - float(dx), float(y) - float(dy)))
                central_access = float(np.clip(1.0 - depot_dist / max(0.55 * float(self.map_size_m), 1.0), 0.0, 1.0))
            else:
                central_access = 0.5
            return float(0.45 * b + 0.16 * g + 0.22 * deg_score + 0.12 * central_access + 0.10 * (1.0 - h) + 1e-6)

        def w_emg(nid: int) -> float:
            b = float(node_attrs[int(nid)].get('builtup_intensity', 0.0))
            g = 1.0 if 'gateway' in str(node_attrs[int(nid)].get('node_role', '')) else 0.0
            h = haz_score(int(nid))
            return float(0.40 * b + 0.30 * g + 0.30 * h + 1e-6)

        def pick(count: int, wfn, excluded: Optional[Set[int]] = None):
            chosen = []
            blocked = set(int(x) for x in (excluded or set()))
            pool = [int(n) for n in nodes if int(n) not in blocked]
            while len(chosen) < int(count) and pool:
                w = np.array([float(max(wfn(n), 1e-9)) for n in pool], dtype=np.float64)
                w = w / max(float(np.sum(w)), 1e-9)
                idx = int(rng.choice(len(pool), p=w))
                n = int(pool.pop(idx))
                x, y = node_xy[n]
                if any(float(np.hypot(x - node_xy[c][0], y - node_xy[c][1])) < float(self.task_min_spacing_m) for c in chosen):
                    continue
                chosen.append(int(n))
            return chosen

        normal_nodes = pick(self.task_normal_count, w_normal)
        emergency_nodes = pick(self.task_emergency_count, w_emg, excluded=set(normal_nodes))

        all_tasks = normal_nodes + emergency_nodes
        haz_share = 0.0
        if all_tasks:
            haz_share = float(sum(1 for n in all_tasks if haz_score(int(n)) >= 0.55) / max(len(all_tasks), 1))
        # cluster balance score
        cid_counts = {}
        for n in all_tasks:
            cid = int(cluster_of.get(int(n), -1))
            cid_counts[cid] = int(cid_counts.get(cid, 0) + 1)
        max_share = float(max(cid_counts.values()) / max(len(all_tasks), 1)) if cid_counts else 1.0
        if len([k for k in cid_counts.keys() if int(k) >= 0]) <= 1:
            balance_score = 1.0
        else:
            balance_score = float(np.clip(1.0 - max(0.0, max_share - 0.35) / 0.65, 0.0, 1.0))

        viol = 0
        for i in range(len(all_tasks) - 1):
            ai = int(all_tasks[i])
            for j in range(i + 1, len(all_tasks)):
                bj = int(all_tasks[j])
                d = float(np.hypot(node_xy[ai][0] - node_xy[bj][0], node_xy[ai][1] - node_xy[bj][1]))
                if d < float(self.task_min_spacing_m):
                    viol += 1

        def pack(nlist: List[int], ttype: str) -> List[Dict[str, Any]]:
            out = []
            for tidx, nid in enumerate(nlist):
                x, y = node_xy[int(nid)]
                out.append({'task_id': f'{ttype}_{tidx}', 'task_type': ttype, 'node_id': int(nid), 'x': float(x), 'y': float(y), 'cluster_id': int(cluster_of.get(int(nid), -1)), 'hazard_exposure': float(haz_score(int(nid)))})
            return out

        return {
            'normal': pack(normal_nodes, 'normal'),
            'emergency': pack(emergency_nodes, 'emergency'),
            'task_cluster_balance_score': float(balance_score),
            'task_hazard_exposure_share': float(haz_share),
            'task_min_spacing_violation_count': int(viol),
        }

    def _compute_realism_bundle(self, node_xy: Dict[int, Tuple[float, float]], node_attrs: Dict[int, Dict[str, Any]], graph: nx.Graph, stats: Dict[str, float], rng: np.random.Generator) -> Dict[str, Any]:
        m = float(self.map_size_m)
        cx0, cx1 = 0.30 * m, 0.70 * m
        c6x0, c6x1 = 0.20 * m, 0.80 * m
        depot = None
        for nid in graph.nodes():
            if str(graph.nodes[int(nid)].get('node_role', 'ordinary')) == 'depot_hub':
                depot = int(nid)
                break
        depot_deg = int(graph.degree(int(depot))) if depot is not None else 0
        depot_in_central_box = False
        depot_in_central_60 = False
        if depot is not None:
            x, y = node_xy[int(depot)]
            depot_in_central_box = bool(cx0 <= x <= cx1 and cx0 <= y <= cx1)
            depot_in_central_60 = bool(c6x0 <= x <= c6x1 and c6x0 <= y <= c6x1)

        major = self._detect_major_clusters(node_attrs=node_attrs, graph=graph)
        major_count = int(len(major))
        major_redundant = 0
        fragile_count = 0
        for comp in major:
            access = []
            for n in comp:
                for nb in graph.neighbors(int(n)):
                    if int(nb) not in comp:
                        access.append((min(int(n), int(nb)), max(int(n), int(nb))))
            access = sorted(set(access))
            if len(access) >= 2:
                major_redundant += 1
            else:
                fragile_count += 1

        v_node, v_gate, v_art = self._count_spacing_violations(node_xy=node_xy, node_attrs=node_attrs)

        # collinear violations on existing triangles (diagnostic count after prune).
        collin_viol = 0
        for b in graph.nodes():
            nbs = [int(n) for n in graph.neighbors(int(b))]
            for i in range(len(nbs) - 1):
                for j in range(i + 1, len(nbs)):
                    a = int(nbs[i]); c = int(nbs[j])
                    if graph.has_edge(min(a, c), max(a, c)) and self._is_near_collinear_triangle(a, int(b), c, node_xy):
                        collin_viol += 1

        hazards = self._generate_hazard_fields(node_xy=node_xy, node_attrs=node_attrs, rng=rng)

        # Grid diagnostics.
        gN = 36
        xs = np.linspace(0.0, m, gN)
        ys = np.linspace(0.0, m, gN)
        rain_core = np.zeros((gN, gN), dtype=np.int8)
        wind_core = np.zeros((gN, gN), dtype=np.int8)
        quake_core = np.zeros((gN, gN), dtype=np.int8)
        lowland = np.zeros((gN, gN), dtype=np.int8)
        qvals = np.zeros((gN, gN), dtype=np.float64)
        for ix, x in enumerate(xs):
            for iy, y in enumerate(ys):
                r = self._ellipse_intensity(float(x), float(y), hazards['rain'])
                w = self._ellipse_intensity(float(x), float(y), hazards['wind'])
                q = self._ellipse_intensity(float(x), float(y), hazards['quake'])
                rain_core[ix, iy] = 1 if r >= 0.70 else 0
                wind_core[ix, iy] = 1 if w >= 0.70 else 0
                quake_core[ix, iy] = 1 if q >= 0.70 else 0
                qvals[ix, iy] = q
                # lowland proxy from nearest node barrier proximity.
                # cheap approximation for realism audit.
                nn = min(node_xy.keys(), key=lambda nid: (node_xy[int(nid)][0] - float(x)) ** 2 + (node_xy[int(nid)][1] - float(y)) ** 2)
                lowland[ix, iy] = 1 if float(node_attrs[int(nn)].get('barrier_proximity', 0.0)) >= 0.50 else 0

        def overlap(a, b):
            inter = float(np.sum((a > 0) & (b > 0)))
            base = float(max(np.sum(a > 0), 1.0))
            return float(inter / base)

        rain_lowland_overlap = overlap(rain_core, lowland)
        ow = overlap(rain_core, wind_core)
        oq = overlap(rain_core, quake_core)
        wq = overlap(wind_core, quake_core)
        max_overlap = float(max(ow, oq, wq))
        wind_shape_ratio = float(max(float(hazards['wind']['major']) / max(float(hazards['wind']['minor']), 1e-6), 1.0))
        gx, gy = np.gradient(qvals)
        smooth = float(np.clip(1.0 - float(np.mean(np.hypot(gx, gy))), 0.0, 1.0))

        tasks = self._sample_tasks_realism(node_xy=node_xy, node_attrs=node_attrs, graph=graph, hazards=hazards, major_clusters=major, rng=rng)

        realism_metrics = {
            'realism_first_pass': 0.0,
            'depot_degree': float(depot_deg),
            'depot_in_central_box': 1.0 if depot_in_central_box else 0.0,
            'major_cluster_count': float(major_count),
            'major_clusters_with_redundant_access_count': float(major_redundant),
            'fragile_cluster_count': float(fragile_count),
            'redundant_cluster_access_rate': float(major_redundant / max(major_count, 1)),
            'min_node_spacing_violation_count': float(v_node),
            'min_gateway_spacing_violation_count': float(v_gate),
            'min_arterial_junction_spacing_violation_count': float(v_art),
            'collinear_triangle_violation_count': float(collin_viol),
            'rain_zone_core_overlap_with_lowland': float(rain_lowland_overlap),
            'wind_swath_shape_ratio': float(wind_shape_ratio),
            'quake_field_smoothness_score': float(smooth),
            'hazard_core_overlap_ratio_max': float(max_overlap),
            'task_cluster_balance_score': float(tasks['task_cluster_balance_score']),
            'task_hazard_exposure_share': float(tasks['task_hazard_exposure_share']),
            'task_min_spacing_violation_count': float(tasks['task_min_spacing_violation_count']),
        }

        realism_pass = bool(
            depot_in_central_60
            and depot_deg >= 2
            and (major_count == 0 or major_redundant >= major_count)
            and v_node == 0
            and v_gate == 0
            and v_art <= 2
            and collin_viol == 0
            and float(stats.get('crossing_fraction', 0.0)) <= 0.02
            and max_overlap <= 0.65
            and float(tasks['task_min_spacing_violation_count']) <= 0
        )
        realism_metrics['realism_first_pass'] = 1.0 if realism_pass else 0.0

        # Penalty to rank candidates in realism-first mode.
        penalty = 0.0
        if not depot_in_central_60:
            penalty += 2.0
        if depot_deg < 2:
            penalty += 1.5
        penalty += 1.2 * max(0.0, float(major_count - major_redundant))
        penalty += 0.4 * float(v_node + v_gate + v_art)
        penalty += 0.3 * float(collin_viol)
        penalty += 1.2 * max(0.0, max_overlap - 0.65)
        penalty += 0.6 * max(0.0, float(tasks['task_min_spacing_violation_count']))

        scene_payload = {
            'depot_node': int(depot) if depot is not None else -1,
            'major_clusters': [sorted(int(n) for n in comp) for comp in major],
            'hazards': hazards,
            'tasks': {
                'normal': list(tasks['normal']),
                'emergency': list(tasks['emergency']),
            },
        }
        return {
            'realism_first_pass': bool(realism_pass),
            'realism_penalty': float(penalty),
            'metrics': realism_metrics,
            'scene_payload': scene_payload,
        }

    def _meso_variant_knobs(self) -> Dict[str, float]:
        """Small, interpretable variant overrides for one-round L calibration."""
        v = str(self.l_map_variant).strip()
        base = {
            'collector_long_links_scale': 1.0,
            'local_diag_prob': 0.35,
            'local_axis_prob': 0.90,
            'local_spacing_scale': 1.0,
            'local_cells_scale': 1.0,
            'local_min_builtup': 0.18,
            'local_merge_radius_m': 70.0,
            'orientation_jitter_local': 0.02,
            'orientation_jitter_arterial_uv': 120.0,
            'orientation_jitter_ring': 0.18,
            'offaxis_max_deg': 24.0,
            'leaf_target_degree': 2.0,
            'high_degree_cap': 4.0,
        }
        if v == 'L_v1a_collector_up_local_down':
            base.update({
                'collector_long_links_scale': 1.35,
                'local_diag_prob': 0.18,
                'local_axis_prob': 0.78,
                'local_spacing_scale': 1.12,
                'local_cells_scale': 0.90,
                'local_min_builtup': 0.21,
            })
        elif v == 'L_v1b_orientation_tighter':
            base.update({
                'orientation_jitter_local': 0.008,
                'orientation_jitter_arterial_uv': 65.0,
                'orientation_jitter_ring': 0.08,
                'local_diag_prob': 0.12,
                'offaxis_max_deg': 18.0,
            })
        elif v == 'L_v1c_abstraction_cleanup':
            base.update({
                'leaf_target_degree': 2.0,
                'high_degree_cap': 4.0,
                'local_merge_radius_m': 85.0,
                'collector_long_links_scale': 1.10,
            })
        return base

    def _apply_variant_a_rebalance(
        self,
        node_xy: Dict[int, Tuple[float, float]],
        node_attrs: Dict[int, Dict[str, Any]],
        graph: nx.Graph,
    ) -> None:
        # Rebalance local->collector share without random edge deletion.
        lengths = []
        total = 0.0
        local_total = 0.0
        collector_total = 0.0
        for u, v, d in graph.edges(data=True):
            l = float(d.get('length_m', d.get('weight', 0.0)))
            rc = str(d.get('road_class', 'collector'))
            total += l
            if rc == 'local':
                local_total += l
                lengths.append((l, int(u), int(v)))
            elif rc == 'collector':
                collector_total += l
        if total <= 1e-9:
            return
        local_share = local_total / total
        if local_share <= 0.60:
            return
        lengths.sort(reverse=True, key=lambda x: float(x[0]))
        # Promote top built-up local connectors to collector class.
        promoted = 0
        for _, u, v in lengths:
            if local_share <= 0.595:
                break
            data = graph.edges[u, v]
            bup = float(data.get('builtup_exposure', 0.0))
            if bup < 0.32:
                continue
            data['road_class'] = 'collector'
            data['travel_speed_kph'] = float(max(float(data.get('travel_speed_kph', 28.0)), 34.0))
            data['lanes'] = int(max(int(data.get('lanes', 1)), 2))
            data['capacity_class'] = 'medium'
            l = float(data.get('length_m', data.get('weight', 0.0)))
            local_total -= l
            collector_total += l
            local_share = local_total / max(total, 1e-9)
            promoted += 1
            if promoted >= 48:
                break

    def _apply_variant_c_cleanup(
        self,
        node_xy: Dict[int, Tuple[float, float]],
        node_attrs: Dict[int, Dict[str, Any]],
        graph: nx.Graph,
    ) -> None:
        # 1) absorb ordinary leaves into nearby collector/local anchors.
        leaves = [int(n) for n in graph.nodes() if graph.degree(int(n)) <= 1 and str(graph.nodes[int(n)].get('node_role', 'ordinary')) == 'ordinary']
        for nid in leaves:
            if nid not in graph:
                continue
            x, y = node_xy[nid]
            cands: List[Tuple[float, int]] = []
            for oid in graph.nodes():
                oid = int(oid)
                if oid == nid or graph.has_edge(nid, oid):
                    continue
                if graph.degree(oid) >= 4:
                    continue
                d = float(np.hypot(x - node_xy[oid][0], y - node_xy[oid][1]))
                if d <= 1100.0:
                    cands.append((d, oid))
            cands.sort(key=lambda t: float(t[0]))
            if not cands:
                continue
            oid = int(cands[0][1])
            if self._edge_crosses_existing(node_xy=node_xy, graph=graph, u=nid, v=oid):
                continue
            l = float(np.hypot(x - node_xy[oid][0], y - node_xy[oid][1]))
            graph.add_edge(
                min(nid, oid),
                max(nid, oid),
                weight=l,
                length_m=l,
                road_class='collector',
                travel_speed_kph=36.0,
                lanes=2,
                capacity_class='medium',
                bridge_or_tunnel=False,
                barrier_exposure=0.0,
                builtup_exposure=float(np.clip(0.5 * (node_attrs[nid].get('builtup_intensity', 0.0) + node_attrs[oid].get('builtup_intensity', 0.0)), 0.0, 1.0)),
                orientation_bin=int(self._orientation_bin(*node_xy[nid], *node_xy[oid])),
            )

        # 2) split super-high-degree collector nodes.
        for nid in list(graph.nodes()):
            nid = int(nid)
            if nid not in graph:
                continue
            if graph.degree(nid) <= 4:
                continue
            role = str(graph.nodes[nid].get('node_role', 'ordinary'))
            if role in {'depot_hub', 'arterial_junction', 'bottleneck'}:
                continue
            nbs = list(graph.neighbors(nid))
            if len(nbs) <= 4:
                continue
            # new nearby junction
            x, y = node_xy[nid]
            nid2 = int(max(node_xy.keys()) + 1)
            nx = float(np.clip(x + 45.0, 0.0, self.map_size_m))
            ny = float(np.clip(y + 35.0, 0.0, self.map_size_m))
            node_xy[nid2] = (nx, ny)
            node_attrs[nid2] = dict(node_attrs[nid])
            node_attrs[nid2]['node_role'] = 'collector_junction'
            graph.add_node(nid2, x=nx, y=ny, **node_attrs[nid2])
            # move farthest neighbors to new node until both <=4
            nbs_sorted = sorted(nbs, key=lambda nb: float(np.hypot(node_xy[int(nb)][0]-x, node_xy[int(nb)][1]-y)), reverse=True)
            for nb in nbs_sorted:
                if graph.degree(nid) <= 4:
                    break
                if graph.has_edge(nid, int(nb)):
                    data = dict(graph.edges[nid, int(nb)])
                    graph.remove_edge(nid, int(nb))
                    l2 = float(np.hypot(node_xy[int(nb)][0]-nx, node_xy[int(nb)][1]-ny))
                    data['weight'] = l2
                    data['length_m'] = l2
                    data['orientation_bin'] = int(self._orientation_bin(*node_xy[int(nb)], nx, ny))
                    graph.add_edge(min(nid2, int(nb)), max(nid2, int(nb)), **data)
            if not graph.has_edge(min(nid, nid2), max(nid, nid2)):
                l = float(np.hypot(x-nx, y-ny))
                graph.add_edge(min(nid, nid2), max(nid, nid2), weight=l, length_m=l, road_class='collector', travel_speed_kph=35.0, lanes=2, capacity_class='medium', bridge_or_tunnel=False, barrier_exposure=0.0, builtup_exposure=float(node_attrs[nid].get('builtup_intensity',0.0)), orientation_bin=int(self._orientation_bin(x,y,nx,ny)))


    def _add_bridge_redundancy_paths(
        self,
        node_xy: Dict[int, Tuple[float, float]],
        node_attrs: Dict[int, Dict[str, Any]],
        graph: nx.Graph,
        max_added_edges: int = 24,
        bridge_side_min_nodes: int = 10,
        max_link_m: float = 2600.0,
    ) -> None:
        # Add backup collector links for bridge-like chokepoints so one broken
        # edge does not disconnect a whole subregion.
        if int(graph.number_of_nodes()) <= 2 or int(graph.number_of_edges()) <= 1:
            return
        budget = int(max(max_added_edges, 0))
        if budget <= 0:
            return

        def _split_if_remove(u: int, v: int) -> Optional[Tuple[set, set]]:
            a, b = (int(u), int(v)) if int(u) < int(v) else (int(v), int(u))
            if not graph.has_edge(a, b):
                return None
            graph.remove_edge(a, b)
            try:
                c1 = set(nx.node_connected_component(graph, a))
                c2 = set(nx.node_connected_component(graph, b))
            except Exception:
                graph.add_edge(a, b)
                return None
            graph.add_edge(a, b)
            if not c1 or not c2:
                return None
            return c1, c2

        scored: List[Tuple[float, int, int, set, set]] = []
        for u, v in nx.bridges(graph):
            sp = _split_if_remove(int(u), int(v))
            if sp is None:
                continue
            c1, c2 = sp
            if min(len(c1), len(c2)) < int(bridge_side_min_nodes):
                continue
            score = float(min(len(c1), len(c2)) * max(len(c1), len(c2)))
            scored.append((score, int(u), int(v), c1, c2))
        scored.sort(key=lambda t: float(t[0]), reverse=True)

        added = 0
        for _sc, _u, _v, ca, cb in scored:
            if added >= budget:
                break
            nodes_a = sorted(list(ca), key=lambda nid: int(graph.degree(int(nid))))[:36]
            nodes_b = sorted(list(cb), key=lambda nid: int(graph.degree(int(nid))))[:36]
            cand: List[Tuple[float, int, int]] = []
            for a in nodes_a:
                if int(graph.degree(int(a))) >= 5:
                    continue
                ra = str(graph.nodes[int(a)].get('node_role', 'ordinary'))
                if ra == 'bottleneck':
                    continue
                xa, ya = node_xy[int(a)]
                for b in nodes_b:
                    if a == b or int(graph.degree(int(b))) >= 5:
                        continue
                    rb = str(graph.nodes[int(b)].get('node_role', 'ordinary'))
                    if rb == 'bottleneck':
                        continue
                    eab = (int(a), int(b)) if int(a) < int(b) else (int(b), int(a))
                    if graph.has_edge(*eab):
                        continue
                    xb, yb = node_xy[int(b)]
                    d = float(np.hypot(xa - xb, ya - yb))
                    if d < 320.0 or d > float(max_link_m):
                        continue
                    if self._edge_crosses_existing(node_xy=node_xy, graph=graph, u=int(a), v=int(b)):
                        continue
                    bprox = 0.5 * float(node_attrs[int(a)].get('barrier_proximity', 0.0)) + 0.5 * float(node_attrs[int(b)].get('barrier_proximity', 0.0))
                    cand.append((float(d + 620.0 * bprox), int(a), int(b)))
            if not cand:
                continue
            cand.sort(key=lambda t: float(t[0]))
            _, a_sel, b_sel = cand[0]
            a, b = (int(a_sel), int(b_sel)) if int(a_sel) < int(b_sel) else (int(b_sel), int(a_sel))
            if graph.has_edge(a, b):
                continue
            length = float(np.hypot(node_xy[a][0] - node_xy[b][0], node_xy[a][1] - node_xy[b][1]))
            graph.add_edge(
                a,
                b,
                weight=length,
                length_m=length,
                road_class='collector',
                travel_speed_kph=40.0,
                lanes=2,
                capacity_class='medium',
                bridge_or_tunnel=False,
                barrier_exposure=float(np.clip(0.5 * (node_attrs[a].get('barrier_proximity', 0.0) + node_attrs[b].get('barrier_proximity', 0.0)), 0.0, 1.0)),
                builtup_exposure=float(np.clip(0.5 * (node_attrs[a].get('builtup_intensity', 0.0) + node_attrs[b].get('builtup_intensity', 0.0)), 0.0, 1.0)),
                orientation_bin=int(self._orientation_bin(*node_xy[a], *node_xy[b])),
            )
            added += 1

    def _build_mesoscopic_candidate(
        self,
        rng: np.random.Generator,
    ) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, Dict[str, Any]], nx.Graph, float, float]:
        map_size = float(self.map_size_m)
        center = np.array([0.5 * map_size, 0.5 * map_size], dtype=np.float64)
        knobs = self._meso_variant_knobs()
        # Hard geometry guards for mesoscopic L:
        # 1) depot should stay near map center (roughly central hub),
        # 2) node placement should respect a minimum spacing floor to avoid near-overlap.
        node_spacing_guard_m = float(max(float(self.l_min_node_spacing_m), min(float(self.min_node_spacing_m), 320.0)))
        depot_center_max_offset_m = float(0.18 * map_size)

        cluster_n = int(rng.integers(5, 8)) if int(self.node_count) >= 380 else int(rng.integers(4, 7))
        clusters: List[Dict[str, float]] = []
        for _ in range(cluster_n):
            clusters.append({
                'cx': float(rng.uniform(0.15 * map_size, 0.85 * map_size)),
                'cy': float(rng.uniform(0.15 * map_size, 0.85 * map_size)),
                'major': float(rng.uniform(1700.0, 2800.0)),
                'minor': float(rng.uniform(900.0, 2200.0)),
                'ang': float(rng.uniform(0.0, np.pi)),
                'w': float(rng.uniform(0.9, 1.3)),
            })
        barrier_n = int(rng.integers(1, 3))
        barriers: List[Dict[str, float]] = []
        for _ in range(barrier_n):
            barriers.append({
                'px': float(rng.uniform(0.2 * map_size, 0.8 * map_size)),
                'py': float(rng.uniform(0.2 * map_size, 0.8 * map_size)),
                'ang': float(rng.uniform(0.0, np.pi)),
                'width': float(rng.uniform(520.0, 980.0)),
            })

        def builtup_field(x: float, y: float) -> float:
            v = 0.0
            for c in clusters:
                dx = float(x - c['cx'])
                dy = float(y - c['cy'])
                ca = float(np.cos(c['ang']))
                sa = float(np.sin(c['ang']))
                u = ca * dx + sa * dy
                w = -sa * dx + ca * dy
                g = float(np.exp(-0.5 * ((u / max(c['major'], 1.0)) ** 2 + (w / max(c['minor'], 1.0)) ** 2)))
                v += float(c['w'] * g)
            return float(np.clip(v / max(1.7, 0.9 * float(len(clusters))), 0.0, 1.0))

        def barrier_field(x: float, y: float) -> float:
            out = 0.0
            for b in barriers:
                ca = float(np.cos(b['ang']))
                sa = float(np.sin(b['ang']))
                nxn = -sa
                nyn = ca
                d = abs((x - b['px']) * nxn + (y - b['py']) * nyn)
                out = max(out, float(np.exp(-((d / max(float(b['width']), 1.0)) ** 2))))
            return float(np.clip(out, 0.0, 1.0))

        crossing_pts: List[Tuple[float, float]] = []
        for b in barriers:
            ca = float(np.cos(b['ang']))
            sa = float(np.sin(b['ang']))
            for _ in range(2):
                t = float(rng.uniform(-0.45 * map_size, 0.45 * map_size))
                x = float(np.clip(b['px'] + t * ca, 0.03 * map_size, 0.97 * map_size))
                y = float(np.clip(b['py'] + t * sa, 0.03 * map_size, 0.97 * map_size))
                crossing_pts.append((x, y))

        node_xy: Dict[int, Tuple[float, float]] = {}
        node_attrs: Dict[int, Dict[str, Any]] = {}
        graph = nx.Graph()

        def role_rank(role: str) -> int:
            ranks = {'ordinary': 1, 'collector_junction': 2, 'area_gateway': 3, 'arterial_junction': 4, 'bottleneck': 5, 'depot_hub': 6}
            return int(ranks.get(str(role), 1))

        def add_node(x: float, y: float, role: str, merge_radius: float = 140.0) -> int:
            x = float(np.clip(x, 0.0, map_size)); y = float(np.clip(y, 0.0, map_size))
            nearest = -1; nearest_d = float('inf')
            for nid, (px, py) in node_xy.items():
                d = float(np.hypot(x - px, y - py))
                if d < nearest_d:
                    nearest_d = d; nearest = int(nid)
            # Role-aware spacing guards (realism-first hard constraints).
            role_thr = float(node_spacing_guard_m)
            rtxt = str(role)
            if 'gateway' in rtxt:
                role_thr = float(max(role_thr, self.l_min_gateway_spacing_m))
            if 'arterial_junction' in rtxt:
                role_thr = float(max(role_thr, self.l_min_arterial_junction_spacing_m))
            merge_thr = float(max(float(max(merge_radius, 1.0)), role_thr))
            if nearest >= 0 and nearest_d <= merge_thr:
                if role_rank(role) > role_rank(str(node_attrs[nearest].get('node_role', 'ordinary'))):
                    node_attrs[nearest]['node_role'] = role
                    graph.nodes[nearest]['node_role'] = role
                return int(nearest)
            nid = int(len(node_xy))
            node_xy[nid] = (x, y)
            node_attrs[nid] = {'node_role': str(role), 'builtup_intensity': float(builtup_field(x, y)), 'barrier_proximity': float(barrier_field(x, y))}
            graph.add_node(nid, x=x, y=y, **node_attrs[nid])
            return nid

        def near_crossing(u: int, v: int, tol: float = 430.0) -> bool:
            mx = 0.5 * (node_xy[u][0] + node_xy[v][0]); my = 0.5 * (node_xy[u][1] + node_xy[v][1])
            return any(float(np.hypot(mx - cx, my - cy)) <= float(tol) for cx, cy in crossing_pts)

        def seg_exposure(u: int, v: int) -> float:
            x1, y1 = node_xy[u]; x2, y2 = node_xy[v]
            vals = []
            for kk in range(5):
                t = float(kk) / 4.0
                vals.append(float(barrier_field((1.0 - t) * x1 + t * x2, (1.0 - t) * y1 + t * y2)))
            return float(max(vals)) if vals else 0.0

        def add_edge(u: int, v: int, road_class: str) -> bool:
            if int(u) == int(v):
                return False
            a, b = (int(u), int(v)) if int(u) < int(v) else (int(v), int(u))
            if graph.has_edge(a, b):
                return False
            role_a = str(node_attrs.get(a, {}).get('node_role', 'ordinary'))
            role_b = str(node_attrs.get(b, {}).get('node_role', 'ordinary'))
            max_da = 5 if role_a in {'arterial_junction', 'depot_hub', 'bottleneck'} else 4
            max_db = 5 if role_b in {'arterial_junction', 'depot_hub', 'bottleneck'} else 4
            if int(graph.degree(a)) >= int(max_da) or int(graph.degree(b)) >= int(max_db):
                return False
            if self._edge_crosses_existing(node_xy=node_xy, graph=graph, u=a, v=b):
                return False
            length = float(np.hypot(node_xy[a][0] - node_xy[b][0], node_xy[a][1] - node_xy[b][1]))
            if length < 120.0:
                return False
            exp = seg_exposure(a, b); bridge = False
            if road_class in {'collector', 'local'} and exp > 0.58:
                return False
            if road_class == 'arterial' and exp > 0.72:
                if not near_crossing(a, b):
                    return False
                bridge = True
                for nid in (a, b):
                    if str(node_attrs[nid].get('node_role', 'ordinary')) == 'ordinary':
                        node_attrs[nid]['node_role'] = 'bottleneck'; graph.nodes[nid]['node_role'] = 'bottleneck'
            speed, lanes, cap = float(rng.uniform(20.0, 35.0)), 1, 'low'
            if road_class == 'arterial':
                speed, lanes, cap = float(rng.uniform(45.0, 70.0)), int(rng.integers(2, 5)), 'high'
            elif road_class == 'collector':
                speed, lanes, cap = float(rng.uniform(30.0, 50.0)), 2, 'medium'
            built_exp = float(0.5 * (node_attrs[a].get('builtup_intensity', 0.0) + node_attrs[b].get('builtup_intensity', 0.0)))
            graph.add_edge(a, b, weight=length, length_m=length, road_class=str(road_class), travel_speed_kph=float(speed), lanes=int(lanes), capacity_class=cap, bridge_or_tunnel=bool(bridge), barrier_exposure=float(exp), builtup_exposure=float(np.clip(built_exp, 0.0, 1.0)), orientation_bin=int(self._orientation_bin(*node_xy[a], *node_xy[b])))
            return True

        # arterial skeleton first: two orientation families
        base_theta = float(rng.uniform(-0.30, 0.30)); ca = float(np.cos(base_theta)); sa = float(np.sin(base_theta))
        def uv_to_xy(u: float, v: float) -> Tuple[float, float]:
            return float(center[0] + ca * u - sa * v), float(center[1] + sa * u + ca * v)
        axis_span = 0.46 * map_size
        line_step = float(rng.uniform(2500.0, 3400.0))
        cluster_uv = []
        for c in clusters:
            dx = float(c['cx'] - center[0]); dy = float(c['cy'] - center[1])
            uu = ca * dx + sa * dy; vv = -sa * dx + ca * dy
            cluster_uv.append((uu, vv))
        v0 = float(np.median([x[1] for x in cluster_uv])); u0 = float(np.median([x[0] for x in cluster_uv]))
        v_offsets = [v0, float(np.clip(v0 - rng.uniform(1800.0, 3000.0), -axis_span, axis_span)), float(np.clip(v0 + rng.uniform(1800.0, 3000.0), -axis_span, axis_span))]
        u_offsets = [u0, float(np.clip(u0 - rng.uniform(1800.0, 3000.0), -axis_span, axis_span)), float(np.clip(u0 + rng.uniform(1800.0, 3000.0), -axis_span, axis_span))]
        line_vals = np.arange(-axis_span, axis_span + 0.5 * line_step, line_step)
        arterial_nodes: List[int] = []
        for v in v_offsets:
            line = []
            for u in line_vals:
                x, y = uv_to_xy(float(u), float(v + rng.uniform(-knobs['orientation_jitter_arterial_uv'], knobs['orientation_jitter_arterial_uv'])))
                if 0.02 * map_size <= x <= 0.98 * map_size and 0.02 * map_size <= y <= 0.98 * map_size:
                    line.append(add_node(x, y, 'arterial_junction', merge_radius=220.0))
            for i in range(len(line) - 1):
                add_edge(line[i], line[i + 1], 'arterial')
            arterial_nodes.extend(line)
        for u in u_offsets:
            line = []
            for v in line_vals:
                x, y = uv_to_xy(float(u + rng.uniform(-knobs['orientation_jitter_arterial_uv'], knobs['orientation_jitter_arterial_uv'])), float(v))
                if 0.02 * map_size <= x <= 0.98 * map_size and 0.02 * map_size <= y <= 0.98 * map_size:
                    line.append(add_node(x, y, 'arterial_junction', merge_radius=220.0))
            for i in range(len(line) - 1):
                add_edge(line[i], line[i + 1], 'arterial')
            arterial_nodes.extend(line)
        arterial_nodes = sorted(set(arterial_nodes))
        if not arterial_nodes:
            n0 = add_node(0.2 * map_size, 0.5 * map_size, 'arterial_junction', 100.0)
            n1 = add_node(0.5 * map_size, 0.5 * map_size, 'arterial_junction', 100.0)
            n2 = add_node(0.8 * map_size, 0.5 * map_size, 'arterial_junction', 100.0)
            add_edge(n0, n1, 'arterial'); add_edge(n1, n2, 'arterial')
            arterial_nodes = [n0, n1, n2]
        # Ensure there is always an arterial anchor close to geometric center.
        center_anchor = int(add_node(float(center[0]), float(center[1]), 'arterial_junction', merge_radius=220.0))
        if center_anchor not in arterial_nodes:
            arterial_nodes.append(center_anchor)
        arterial_nodes = sorted(set(arterial_nodes))

        # depot hub: not fixed at geometric center.
        # Prefer a center-near arterial node with at least moderate branching
        # so depot can route to multi-directions from the very beginning.
        # Central-box first depot candidate policy:
        # prefer central 40% x 40%; fallback central 60% x 60%.
        c40_min = 0.30 * map_size
        c40_max = 0.70 * map_size
        c60_min = 0.20 * map_size
        c60_max = 0.80 * map_size
        core_art_40 = []
        core_art_60 = []
        for nid in arterial_nodes:
            x, y = node_xy[int(nid)]
            if c60_min <= x <= c60_max and c60_min <= y <= c60_max:
                core_art_60.append(int(nid))
                if c40_min <= x <= c40_max and c40_min <= y <= c40_max:
                    core_art_40.append(int(nid))
        dep_pool = core_art_40 if core_art_40 else (core_art_60 if core_art_60 else list(arterial_nodes))

        dep_cands = []
        for nid in dep_pool:
            x, y = node_xy[nid]
            bup = float(node_attrs[nid].get('builtup_intensity', 0.0))
            dcenter = float(np.hypot(x - center[0], y - center[1]))
            deg_penalty = float(max(0, 2 - int(graph.degree(int(nid)))))
            # Prefer candidates that can naturally fan-out to multiple directions.
            sector_bins = set()
            for oid in arterial_nodes:
                oid = int(oid)
                if oid == int(nid):
                    continue
                ox, oy = node_xy[int(oid)]
                dist = float(np.hypot(ox - x, oy - y))
                if dist < 700.0 or dist > 0.50 * map_size:
                    continue
                ang = float(np.arctan2(oy - y, ox - x))
                aa = (ang + 2.0 * np.pi) % (2.0 * np.pi)
                sector_bins.add(int(aa / (np.pi / 2.0)))
            directional_bonus = float(min(len(sector_bins), 4))
            score = float(
                abs(bup - 0.5)
                + 0.55 * dcenter / max(0.5 * map_size, 1.0)
                + 0.20 * deg_penalty
                - 0.16 * directional_bonus
            )
            dep_cands.append((score, nid))
        dep_cands.sort(key=lambda t: float(t[0]))
        depot_id = int(dep_cands[0][1]) if dep_cands else int(center_anchor)
        # Final hard guard: depot must stay in central 60% box.
        dep_x, dep_y = node_xy[int(depot_id)]
        if not (c60_min <= dep_x <= c60_max and c60_min <= dep_y <= c60_max):
            depot_id = int(center_anchor)
        node_attrs[depot_id]['node_role'] = 'depot_hub'; graph.nodes[depot_id]['node_role'] = 'depot_hub'

        collector_nodes: List[int] = []
        for c in clusters:
            cx, cy = float(c['cx']), float(c['cy'])
            gtw = add_node(cx, cy, 'area_gateway', merge_radius=180.0)
            nearest_art = min(arterial_nodes, key=lambda nid: float(np.hypot(node_xy[nid][0] - cx, node_xy[nid][1] - cy)))
            add_edge(gtw, int(nearest_art), 'collector')
            ring_n = int(rng.integers(10, 15)); ring_r = float(rng.uniform(1500.0, 2600.0))
            ring = []
            for i in range(ring_n):
                ang = float(c['ang'] + 2.0 * np.pi * i / ring_n + rng.uniform(-knobs['orientation_jitter_ring'], knobs['orientation_jitter_ring']))
                x = cx + ring_r * np.cos(ang); y = cy + ring_r * np.sin(ang)
                if 0.02 * map_size <= x <= 0.98 * map_size and 0.02 * map_size <= y <= 0.98 * map_size:
                    rr = add_node(x, y, 'collector_junction', merge_radius=160.0)
                    ring.append(rr); collector_nodes.append(rr)
            for i in range(len(ring)):
                add_edge(ring[i], ring[(i + 1) % len(ring)], 'collector')
            step = max(1, len(ring) // 3)
            for rr in ring[::step]:
                add_edge(gtw, rr, 'collector')


        # Ensure depot has 3~4 directional exits toward surrounding network.
        # This avoids a single-corridor depot bottleneck and improves early exploration.
        desired_depot_exits = 4 if int(self.node_count) >= 340 else 3
        desired_direction_bins = 3 if int(self.node_count) >= 340 else 2
        dep_xy = node_xy[int(depot_id)]

        def _ang_diff(a: float, b: float) -> float:
            d = abs(float(a) - float(b))
            while d > np.pi:
                d -= np.pi
            return float(min(d, np.pi - d))

        def _dir_bins_count(angles: List[float]) -> int:
            bins = set()
            for a in angles:
                aa = (float(a) + 2.0 * np.pi) % (2.0 * np.pi)
                bins.add(int(aa / (np.pi / 4.0)))
            return int(len(bins))

        target_dirs = [
            float(base_theta),
            float(base_theta + 0.5 * np.pi),
            float(base_theta + np.pi),
            float(base_theta + 1.5 * np.pi),
        ]

        def _depot_neighbor_angles() -> List[float]:
            out: List[float] = []
            for nb in graph.neighbors(int(depot_id)):
                x, y = node_xy[int(nb)]
                out.append(float(np.arctan2(y - dep_xy[1], x - dep_xy[0])))
            return out

        def _candidate_pool() -> List[Tuple[float, float, float, int]]:
            cands: List[Tuple[float, float, float, int]] = []
            for nid in graph.nodes():
                nid = int(nid)
                if nid == int(depot_id):
                    continue
                role = str(graph.nodes[nid].get('node_role', 'ordinary'))
                if role not in {'arterial_junction', 'area_gateway', 'collector_junction', 'bottleneck'}:
                    continue
                if graph.has_edge(min(int(depot_id), nid), max(int(depot_id), nid)):
                    continue
                x, y = node_xy[nid]
                dx = float(x - dep_xy[0]); dy = float(y - dep_xy[1])
                dist = float(np.hypot(dx, dy))
                if dist < 650.0 or dist > 0.46 * map_size:
                    continue
                ang = float(np.arctan2(dy, dx))
                exp = float(seg_exposure(int(depot_id), int(nid)))
                cands.append((dist, ang, exp, int(nid)))
            return cands

        if int(graph.degree(int(depot_id))) < int(desired_depot_exits) or _dir_bins_count(_depot_neighbor_angles()) < int(desired_direction_bins):
            pool = _candidate_pool()
            selected_angles: List[float] = list(_depot_neighbor_angles())

            # Pass-1: choose directional candidates around primary axes.
            for td in target_dirs:
                if int(graph.degree(int(depot_id))) >= int(desired_depot_exits) and _dir_bins_count(selected_angles) >= int(desired_direction_bins):
                    break
                ranked = sorted(
                    pool,
                    key=lambda it: float(3.2 * _ang_diff(it[1], td) + 0.55 * (it[0] / max(0.46 * map_size, 1.0)) + 0.60 * it[2]),
                )
                for dist, ang, _exp, nid in ranked[:24]:
                    if graph.has_edge(min(int(depot_id), int(nid)), max(int(depot_id), int(nid))):
                        continue
                    if selected_angles and min(_ang_diff(float(ang), a0) for a0 in selected_angles) < np.deg2rad(18.0):
                        continue
                    road_cls = 'arterial' if float(dist) >= 1700.0 else 'collector'
                    if add_edge(int(depot_id), int(nid), road_cls):
                        selected_angles.append(float(ang))
                        break

            # Pass-2: fill remaining exits with nearest feasible diverse directions.
            if int(graph.degree(int(depot_id))) < int(desired_depot_exits) or _dir_bins_count(selected_angles) < int(desired_direction_bins):
                ranked2 = sorted(pool, key=lambda it: float(it[0] + 900.0 * it[2]))
                for dist, ang, _exp, nid in ranked2:
                    if int(graph.degree(int(depot_id))) >= int(desired_depot_exits) and _dir_bins_count(selected_angles) >= int(desired_direction_bins):
                        break
                    if graph.has_edge(min(int(depot_id), int(nid)), max(int(depot_id), int(nid))):
                        continue
                    if selected_angles and min(_ang_diff(float(ang), a0) for a0 in selected_angles) < np.deg2rad(28.0):
                        continue
                    road_cls = 'arterial' if float(dist) >= 1500.0 else 'collector'
                    if add_edge(int(depot_id), int(nid), road_cls):
                        selected_angles.append(float(ang))

            # Pass-3: if direction coverage is still poor, create directional anchor spokes.
            if _dir_bins_count(selected_angles) < int(desired_direction_bins):
                for td in target_dirs:
                    if _dir_bins_count(selected_angles) >= int(desired_direction_bins):
                        break
                    if selected_angles and min(_ang_diff(float(td), a0) for a0 in selected_angles) < np.deg2rad(30.0):
                        continue
                    r = float(rng.uniform(850.0, 1200.0))
                    sx = float(np.clip(dep_xy[0] + r * np.cos(float(td)), 0.04 * map_size, 0.96 * map_size))
                    sy = float(np.clip(dep_xy[1] + r * np.sin(float(td)), 0.04 * map_size, 0.96 * map_size))
                    spoke_id = int(add_node(sx, sy, 'area_gateway', merge_radius=20.0))
                    if int(spoke_id) == int(depot_id):
                        continue
                    eab = (min(int(depot_id), int(spoke_id)), max(int(depot_id), int(spoke_id)))
                    if not graph.has_edge(*eab):
                        ok_spoke = add_edge(int(depot_id), int(spoke_id), 'collector')
                        if (not ok_spoke) and (not graph.has_edge(*eab)):
                            # Hard fallback near depot: keep short directional exits even
                            # if local crossing heuristic is conservative at hub area.
                            a, b = eab
                            length = float(np.hypot(node_xy[a][0] - node_xy[b][0], node_xy[a][1] - node_xy[b][1]))
                            graph.add_edge(
                                a,
                                b,
                                weight=length,
                                length_m=length,
                                road_class='collector',
                                travel_speed_kph=38.0,
                                lanes=2,
                                capacity_class='medium',
                                bridge_or_tunnel=False,
                                barrier_exposure=float(seg_exposure(a, b)),
                                builtup_exposure=float(np.clip(0.5 * (node_attrs[a].get('builtup_intensity', 0.0) + node_attrs[b].get('builtup_intensity', 0.0)), 0.0, 1.0)),
                                orientation_bin=int(self._orientation_bin(*node_xy[a], *node_xy[b])),
                            )
                    # Connect spoke to nearby network in similar direction.
                    sx0, sy0 = node_xy[int(spoke_id)]
                    spoke_cands: List[Tuple[float, int]] = []
                    for oid in graph.nodes():
                        oid = int(oid)
                        if oid in {int(depot_id), int(spoke_id)}:
                            continue
                        role = str(graph.nodes[oid].get('node_role', 'ordinary'))
                        if role not in {'arterial_junction', 'area_gateway', 'collector_junction', 'bottleneck'}:
                            continue
                        if graph.has_edge(min(int(spoke_id), oid), max(int(spoke_id), oid)):
                            continue
                        ox, oy = node_xy[oid]
                        dist2 = float(np.hypot(ox - sx0, oy - sy0))
                        if dist2 < 500.0 or dist2 > 3000.0:
                            continue
                        ang2 = float(np.arctan2(oy - dep_xy[1], ox - dep_xy[0]))
                        if _ang_diff(ang2, float(td)) > np.deg2rad(55.0):
                            continue
                        spoke_cands.append((dist2, oid))
                    spoke_cands.sort(key=lambda x: float(x[0]))
                    for dist2, oid in spoke_cands[:12]:
                        road_cls2 = 'arterial' if float(dist2) >= 1500.0 else 'collector'
                        if add_edge(int(spoke_id), int(oid), road_cls2):
                            break
                    sx1, sy1 = node_xy[int(spoke_id)]
                    selected_angles.append(float(np.arctan2(sy1 - dep_xy[1], sx1 - dep_xy[0])))

        local_nodes: List[int] = []
        for c in clusters:
            cx, cy, ang_raw = float(c['cx']), float(c['cy']), float(c['ang'])
            cand_a = [base_theta, base_theta + 0.5 * np.pi]
            ang = min(cand_a, key=lambda aa: abs(((ang_raw - aa + np.pi) % np.pi) - 0.5 * np.pi))
            ang = float(ang + rng.uniform(-knobs['orientation_jitter_local'], knobs['orientation_jitter_local']))
            ca_l, sa_l = float(np.cos(ang)), float(np.sin(ang))
            nx_base = (int(rng.integers(16, 23)), int(rng.integers(16, 23))) if int(self.node_count) >= 380 else (int(rng.integers(14, 20)), int(rng.integers(14, 20)))
            nx_cells, ny_cells = max(8, int(round(nx_base[0] * knobs['local_cells_scale']))), max(8, int(round(nx_base[1] * knobs['local_cells_scale'])))
            spacing = float(rng.uniform(650.0, 1200.0) * knobs['local_spacing_scale'])
            idx: Dict[Tuple[int, int], int] = {}
            for ix in range(nx_cells):
                for iy in range(ny_cells):
                    uu = (ix - 0.5 * (nx_cells - 1)) * spacing + float(rng.uniform(-60.0, 60.0))
                    vv = (iy - 0.5 * (ny_cells - 1)) * spacing + float(rng.uniform(-60.0, 60.0))
                    x = cx + ca_l * uu - sa_l * vv; y = cy + sa_l * uu + ca_l * vv
                    if not (0.03 * map_size <= x <= 0.97 * map_size and 0.03 * map_size <= y <= 0.97 * map_size):
                        continue
                    if builtup_field(x, y) < float(knobs['local_min_builtup']) or barrier_field(x, y) > 0.82:
                        continue
                    nid = add_node(x, y, 'ordinary', merge_radius=float(knobs['local_merge_radius_m']))
                    idx[(ix, iy)] = nid; local_nodes.append(nid)
            for ix in range(nx_cells):
                for iy in range(ny_cells):
                    cur = idx.get((ix, iy))
                    if cur is None:
                        continue
                    r = idx.get((ix + 1, iy)); d = idx.get((ix, iy + 1))
                    if r is not None and float(rng.uniform(0.0, 1.0)) <= float(knobs['local_axis_prob']):
                        add_edge(cur, r, 'local')
                    if d is not None and float(rng.uniform(0.0, 1.0)) <= float(knobs['local_axis_prob']):
                        add_edge(cur, d, 'local')
                    diag = idx.get((ix + 1, iy + 1))
                    if diag is not None and float(rng.uniform(0.0, 1.0)) <= float(knobs['local_diag_prob']):
                        add_edge(cur, diag, 'local')

            # collector fill-in long links inside built-up patches
            patch_nodes = sorted(set(idx.values()))
            if len(patch_nodes) >= 14:
                patch_nodes = sorted(
                    patch_nodes,
                    key=lambda nid: float(node_xy[int(nid)][0] + node_xy[int(nid)][1]),
                )
                jump = max(4, len(patch_nodes) // 5)
                max_links = int(max(8, round(24 * float(knobs['collector_long_links_scale']))))
                for kk in range(0, min(len(patch_nodes) - jump, max_links), 2):
                    a = int(patch_nodes[kk])
                    b = int(patch_nodes[kk + jump])
                    add_edge(a, b, 'collector')

        local_pool = sorted(set(local_nodes))
        for nid in local_pool:
            if nid not in graph:
                continue
            if graph.degree(nid) >= 3:
                continue
            x, y = node_xy[nid]
            cand = []
            for oid in local_pool:
                if oid == nid or oid not in graph:
                    continue
                dxy = float(np.hypot(x - node_xy[oid][0], y - node_xy[oid][1]))
                if dxy <= 1500.0:
                    cand.append((dxy, oid))
            cand.sort(key=lambda t: float(t[0]))
            for _, oid in cand[:4]:
                if graph.degree(nid) >= 3:
                    break
                add_edge(nid, int(oid), 'local')

        collector_pool = [n for n in graph.nodes() if graph.nodes[n].get('node_role') in {'collector_junction', 'area_gateway', 'arterial_junction', 'depot_hub'}]
        for nid in sorted(set(local_nodes)):
            if graph.degree(nid) >= 3:
                continue
            x, y = node_xy[nid]
            dlist = []
            for cid in collector_pool:
                if cid == nid:
                    continue
                dd = float(np.hypot(x - node_xy[cid][0], y - node_xy[cid][1]))
                if dd <= 950.0:
                    dlist.append((dd, cid))
            dlist.sort(key=lambda t: float(t[0]))
            if dlist:
                add_edge(nid, int(dlist[0][1]), 'collector')

        # reduce leafs and increase serviceable branching in mesoscopic graph
        for nid in list(graph.nodes()):
            role = str(graph.nodes[nid].get('node_role', 'ordinary'))
            if role in {'depot_hub', 'arterial_junction', 'bottleneck'}:
                continue
            x, y = node_xy[int(nid)]
            while graph.degree(int(nid)) < 2:
                cand2 = []
                for oid in graph.nodes():
                    if int(oid) == int(nid) or graph.has_edge(int(nid), int(oid)):
                        continue
                    d2 = float(np.hypot(x - node_xy[int(oid)][0], y - node_xy[int(oid)][1]))
                    if d2 <= 1800.0:
                        cand2.append((d2, int(oid)))
                cand2.sort(key=lambda t: float(t[0]))
                if not cand2:
                    break
                target_oid = int(cand2[0][1])
                cls = 'local' if role == 'ordinary' else 'collector'
                if not add_edge(int(nid), target_oid, cls):
                    a, b = (int(nid), target_oid) if int(nid) < target_oid else (target_oid, int(nid))
                    if not graph.has_edge(a, b):
                        if self._edge_crosses_existing(node_xy=node_xy, graph=graph, u=a, v=b):
                            break
                        length = float(np.hypot(node_xy[a][0] - node_xy[b][0], node_xy[a][1] - node_xy[b][1]))
                        graph.add_edge(
                            a,
                            b,
                            weight=length,
                            length_m=length,
                            road_class=cls,
                            travel_speed_kph=30.0 if cls == 'local' else 40.0,
                            lanes=1 if cls == 'local' else 2,
                            capacity_class='low' if cls == 'local' else 'medium',
                            bridge_or_tunnel=False,
                            barrier_exposure=float(seg_exposure(a, b)),
                            builtup_exposure=float(np.clip(0.5 * (node_attrs[a].get('builtup_intensity', 0.0) + node_attrs[b].get('builtup_intensity', 0.0)), 0.0, 1.0)),
                            orientation_bin=int(self._orientation_bin(*node_xy[a], *node_xy[b])),
                        )
                    else:
                        break

        comps = [set(c) for c in nx.connected_components(graph)]
        guard = 0
        while len(comps) > 1:
            guard += 1
            if guard > 2000:
                break
            c1, c2 = comps[0], comps[1]
            best = None; best_d = float('inf')
            for a in c1:
                for b in c2:
                    d = float(np.hypot(node_xy[a][0] - node_xy[b][0], node_xy[a][1] - node_xy[b][1]))
                    if d < best_d:
                        best_d = d; best = (int(a), int(b))
            if best is None:
                break
            cls = 'collector' if best_d <= 2000.0 else 'arterial'
            ok = add_edge(best[0], best[1], cls)
            if not ok:
                a, b = (best[0], best[1]) if best[0] < best[1] else (best[1], best[0])
                length = float(np.hypot(node_xy[a][0] - node_xy[b][0], node_xy[a][1] - node_xy[b][1]))
                if not graph.has_edge(a, b):
                    graph.add_edge(
                        a,
                        b,
                        weight=length,
                        length_m=length,
                        road_class=cls,
                        travel_speed_kph=35.0 if cls == 'collector' else 55.0,
                        lanes=2,
                        capacity_class='medium' if cls == 'collector' else 'high',
                        bridge_or_tunnel=False,
                        barrier_exposure=float(seg_exposure(a, b)),
                        builtup_exposure=float(np.clip(0.5 * (node_attrs[a].get('builtup_intensity', 0.0) + node_attrs[b].get('builtup_intensity', 0.0)), 0.0, 1.0)),
                        orientation_bin=int(self._orientation_bin(*node_xy[a], *node_xy[b])),
                    )
            comps = [set(c) for c in nx.connected_components(graph)]


        # Final depot egress coverage check after local/connectivity fill-in.
        if int(depot_id) in graph:
            dep_xy = node_xy[int(depot_id)]

            def _dep_angles_now() -> List[float]:
                vals: List[float] = []
                for nb in graph.neighbors(int(depot_id)):
                    x, y = node_xy[int(nb)]
                    vals.append(float(np.arctan2(y - dep_xy[1], x - dep_xy[0])))
                return vals

            def _dep_bins_now(angles: List[float]) -> int:
                bins = set()
                for a in angles:
                    aa = (float(a) + 2.0 * np.pi) % (2.0 * np.pi)
                    bins.add(int(aa / (np.pi / 4.0)))
                return int(len(bins))

            target_bin_goal = 3
            for td in target_dirs:
                if _dep_bins_now(_dep_angles_now()) >= int(target_bin_goal):
                    break
                if _dep_angles_now() and min(_ang_diff(float(td), a0) for a0 in _dep_angles_now()) < np.deg2rad(30.0):
                    continue

                # Try direct depot -> network link first.
                cands: List[Tuple[float, int]] = []
                for oid in graph.nodes():
                    oid = int(oid)
                    if oid == int(depot_id) or graph.has_edge(min(int(depot_id), oid), max(int(depot_id), oid)):
                        continue
                    role = str(graph.nodes[oid].get('node_role', 'ordinary'))
                    if role not in {'arterial_junction', 'area_gateway', 'collector_junction', 'bottleneck'}:
                        continue
                    ox, oy = node_xy[oid]
                    dist = float(np.hypot(ox - dep_xy[0], oy - dep_xy[1]))
                    if dist < 700.0 or dist > 0.50 * map_size:
                        continue
                    ang = float(np.arctan2(oy - dep_xy[1], ox - dep_xy[0]))
                    if _ang_diff(ang, float(td)) > np.deg2rad(55.0):
                        continue
                    cands.append((dist + 800.0 * float(seg_exposure(int(depot_id), int(oid))), int(oid)))
                cands.sort(key=lambda x: float(x[0]))

                added = False
                for _score, oid in cands[:24]:
                    dxy = float(np.hypot(node_xy[int(oid)][0] - dep_xy[0], node_xy[int(oid)][1] - dep_xy[1]))
                    road_cls = 'arterial' if dxy >= 1600.0 else 'collector'
                    if add_edge(int(depot_id), int(oid), road_cls):
                        added = True
                        break

                if added:
                    continue

                # Hard fallback: create a directional spoke anchor from depot.
                r = float(rng.uniform(900.0, 1200.0))
                sx = float(np.clip(dep_xy[0] + r * np.cos(float(td)), 0.04 * map_size, 0.96 * map_size))
                sy = float(np.clip(dep_xy[1] + r * np.sin(float(td)), 0.04 * map_size, 0.96 * map_size))
                spoke_id = int(add_node(sx, sy, 'area_gateway', merge_radius=1.0))
                if int(spoke_id) == int(depot_id):
                    continue
                a, b = (int(depot_id), int(spoke_id)) if int(depot_id) < int(spoke_id) else (int(spoke_id), int(depot_id))
                if not graph.has_edge(a, b):
                    length = float(np.hypot(node_xy[a][0] - node_xy[b][0], node_xy[a][1] - node_xy[b][1]))
                    graph.add_edge(
                        a,
                        b,
                        weight=length,
                        length_m=length,
                        road_class='collector',
                        travel_speed_kph=38.0,
                        lanes=2,
                        capacity_class='medium',
                        bridge_or_tunnel=False,
                        barrier_exposure=float(seg_exposure(a, b)),
                        builtup_exposure=float(np.clip(0.5 * (node_attrs[a].get('builtup_intensity', 0.0) + node_attrs[b].get('builtup_intensity', 0.0)), 0.0, 1.0)),
                        orientation_bin=int(self._orientation_bin(*node_xy[a], *node_xy[b])),
                    )

                # Try to attach spoke to nearby network.
                sx0, sy0 = node_xy[int(spoke_id)]
                spoke_cands: List[Tuple[float, int]] = []
                for oid in graph.nodes():
                    oid = int(oid)
                    if oid in {int(depot_id), int(spoke_id)}:
                        continue
                    role = str(graph.nodes[oid].get('node_role', 'ordinary'))
                    if role not in {'arterial_junction', 'area_gateway', 'collector_junction', 'bottleneck'}:
                        continue
                    if graph.has_edge(min(int(spoke_id), oid), max(int(spoke_id), oid)):
                        continue
                    ox, oy = node_xy[oid]
                    dist2 = float(np.hypot(ox - sx0, oy - sy0))
                    if dist2 < 500.0 or dist2 > 2800.0:
                        continue
                    ang2 = float(np.arctan2(oy - dep_xy[1], ox - dep_xy[0]))
                    if _ang_diff(ang2, float(td)) > np.deg2rad(60.0):
                        continue
                    spoke_cands.append((dist2 + 600.0 * float(seg_exposure(int(spoke_id), int(oid))), int(oid)))
                spoke_cands.sort(key=lambda x: float(x[0]))
                for _sc, oid in spoke_cands[:18]:
                    cls2 = 'arterial' if float(np.hypot(node_xy[int(oid)][0] - sx0, node_xy[int(oid)][1] - sy0)) >= 1500.0 else 'collector'
                    if add_edge(int(spoke_id), int(oid), cls2):
                        break

        if str(self.l_map_variant).strip() == 'L_v1a_collector_up_local_down':
            self._apply_variant_a_rebalance(node_xy=node_xy, node_attrs=node_attrs, graph=graph)
        if str(self.l_map_variant).strip() == 'L_v1c_abstraction_cleanup':
            self._apply_variant_c_cleanup(node_xy=node_xy, node_attrs=node_attrs, graph=graph)

        self._contract_decision_chains(node_xy=node_xy, node_attrs=node_attrs, graph=graph)
        self._add_bridge_redundancy_paths(
            node_xy=node_xy,
            node_attrs=node_attrs,
            graph=graph,
            max_added_edges=26,
            bridge_side_min_nodes=10,
            max_link_m=2600.0,
        )
        node_xy, node_attrs, graph = self._reindex_graph(node_xy=node_xy, node_attrs=node_attrs, graph=graph)

        builtup_frac = self._estimate_field_area_fraction(field_fn=builtup_field, threshold=0.10, samples_per_axis=54)
        barrier_frac = self._estimate_field_area_fraction(field_fn=barrier_field, threshold=0.48, samples_per_axis=54)
        return node_xy, node_attrs, graph, float(builtup_frac), float(barrier_frac)

    def _contract_decision_chains(self, node_xy: Dict[int, Tuple[float, float]], node_attrs: Dict[int, Dict[str, Any]], graph: nx.Graph) -> None:
        changed = True
        while changed:
            changed = False
            for nid in list(graph.nodes()):
                role = str(graph.nodes[nid].get('node_role', 'ordinary'))
                if role in {'depot_hub', 'arterial_junction', 'area_gateway', 'bottleneck'}:
                    continue
                if graph.degree(nid) != 2:
                    continue
                nbs = list(graph.neighbors(nid))
                if len(nbs) != 2:
                    continue
                a, b = int(nbs[0]), int(nbs[1])
                if a == b or graph.has_edge(a, b):
                    continue
                e1 = graph.edges[nid, a]; e2 = graph.edges[nid, b]
                rc1 = str(e1.get('road_class', 'local')); rc2 = str(e2.get('road_class', 'local'))
                if 'arterial' in {rc1, rc2}:
                    continue
                rc_keep = 'collector' if ('collector' in {rc1, rc2}) else 'local'
                lsum = float(e1.get('length_m', e1.get('weight', 0.0)) + e2.get('length_m', e2.get('weight', 0.0)))
                if lsum < 350.0:
                    continue
                if self._edge_crosses_existing(node_xy=node_xy, graph=graph, u=a, v=b, ignore_nodes={nid}):
                    continue
                graph.remove_node(nid)
                node_xy.pop(int(nid), None); node_attrs.pop(int(nid), None)
                graph.add_edge(int(min(a, b)), int(max(a, b)), weight=float(lsum), length_m=float(lsum), road_class=rc_keep, travel_speed_kph=float(0.5 * (float(e1.get('travel_speed_kph', 30.0)) + float(e2.get('travel_speed_kph', 30.0)))), lanes=int(max(int(e1.get('lanes', 1)), int(e2.get('lanes', 1)))), capacity_class=str(e1.get('capacity_class', e2.get('capacity_class', 'medium'))), bridge_or_tunnel=bool(e1.get('bridge_or_tunnel', False) or e2.get('bridge_or_tunnel', False)), barrier_exposure=float(max(float(e1.get('barrier_exposure', 0.0)), float(e2.get('barrier_exposure', 0.0)))), builtup_exposure=float(np.clip(0.5 * (float(graph.nodes[a].get('builtup_intensity', 0.0)) + float(graph.nodes[b].get('builtup_intensity', 0.0))), 0.0, 1.0)), orientation_bin=int(self._orientation_bin(*node_xy[a], *node_xy[b])))
                changed = True
                break

    def _reindex_graph(self, node_xy: Dict[int, Tuple[float, float]], node_attrs: Dict[int, Dict[str, Any]], graph: nx.Graph) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, Dict[str, Any]], nx.Graph]:
        order = sorted(int(n) for n in graph.nodes())
        # Keep depot_hub mapped to node-0 after reindex because env/planner use
        # node-0 as global depot anchor for routing/reload/recovery semantics.
        depot_old: Optional[int] = None
        for n in order:
            role = str(graph.nodes[int(n)].get("node_role", "ordinary"))
            if role == "depot_hub":
                depot_old = int(n)
                break
        if depot_old is not None:
            order = [int(depot_old)] + [int(n) for n in order if int(n) != int(depot_old)]
        old2new = {old: idx for idx, old in enumerate(order)}
        new_xy: Dict[int, Tuple[float, float]] = {}
        new_attrs: Dict[int, Dict[str, Any]] = {}
        new_graph = nx.Graph()
        for old, new in old2new.items():
            x, y = node_xy[int(old)]
            attrs = dict(node_attrs.get(int(old), {}))
            new_xy[int(new)] = (float(x), float(y))
            new_attrs[int(new)] = attrs
            new_graph.add_node(int(new), x=float(x), y=float(y), **attrs)
        for u, v, data in graph.edges(data=True):
            nu = int(old2new[int(u)])
            nv = int(old2new[int(v)])
            if nu == nv:
                continue
            a, b = (nu, nv) if nu < nv else (nv, nu)
            new_graph.add_edge(a, b, **dict(data))
        return new_xy, new_attrs, new_graph

    def _compute_map_stats(self) -> Dict[str, float]:
        return self._compute_map_stats_from(node_xy=self.node_xy, graph=self.graph, builtup_area_fraction=0.0, barrier_area_fraction=0.0)

    def _compute_map_stats_from(self, node_xy: Dict[int, Tuple[float, float]], graph: nx.Graph, builtup_area_fraction: float, barrier_area_fraction: float) -> Dict[str, float]:
        n = int(graph.number_of_nodes()); m = int(graph.number_of_edges())
        deg = np.array([float(graph.degree(v)) for v in graph.nodes()], dtype=np.float64) if n > 0 else np.zeros((0,), dtype=np.float64)
        lengths = []
        l_art, l_col, l_loc = 0.0, 0.0, 0.0
        angles = []
        for u, v, data in graph.edges(data=True):
            length = float(data.get('length_m', data.get('weight', np.hypot(node_xy[int(u)][0] - node_xy[int(v)][0], node_xy[int(u)][1] - node_xy[int(v)][1]))))
            lengths.append(length)
            rc = str(data.get('road_class', 'collector'))
            if rc == 'arterial': l_art += length
            elif rc == 'collector': l_col += length
            else: l_loc += length
            theta = float(np.arctan2(node_xy[int(v)][1] - node_xy[int(u)][1], node_xy[int(v)][0] - node_xy[int(u)][0]))
            if theta < 0.0: theta += np.pi
            if theta >= np.pi: theta -= np.pi
            angles.append(theta)
        arr = np.array(lengths, dtype=np.float64) if lengths else np.zeros((0,), dtype=np.float64)
        nonlocal_lengths = np.array([float(data.get('length_m', data.get('weight', 0.0))) for _, _, data in graph.edges(data=True) if str(data.get('road_class', 'collector')) != 'local'], dtype=np.float64)
        if nonlocal_lengths.size <= 0:
            nonlocal_lengths = arr.copy()
        total_len = float(max(float(np.sum(arr)), 1e-9))
        crossing_fraction = self._compute_crossing_fraction(node_xy=node_xy, graph=graph)
        off_axis = 0.0; main_modes = 0.0
        if angles:
            ori = np.array(angles, dtype=np.float64)
            hist, bins = np.histogram(ori, bins=np.linspace(0.0, np.pi, 13))
            shares = hist / max(int(np.sum(hist)), 1)
            top = np.argsort(hist)[-2:]
            mode_angles = [float(0.5 * (bins[int(b)] + bins[int(b) + 1])) for b in top]
            main_modes = float(2 if float(np.sort(shares)[-2]) >= 0.12 else 1)
            cnt = 0
            for t in ori:
                dmin = min(min(abs(float(t) - ma), np.pi - abs(float(t) - ma)) for ma in mode_angles)
                if dmin > np.deg2rad(24.0):
                    cnt += 1
            off_axis = float(cnt / max(len(ori), 1))
        stats = {
            'num_nodes': float(n), 'num_edges': float(m), 'avg_degree': float((2.0 * m / n) if n > 0 else 0.0),
            'median_edge_length_m': float(np.median(arr)) if arr.size else 0.0,
            'p90_edge_length_m': float(np.quantile(nonlocal_lengths, 0.90)) if nonlocal_lengths.size else 0.0,
            'leaf_fraction': float(np.mean(deg <= 1.0)) if deg.size else 0.0,
            'deg3_fraction': float(np.mean(np.isclose(deg, 3.0))) if deg.size else 0.0,
            'deg4_fraction': float(np.mean(np.isclose(deg, 4.0))) if deg.size else 0.0,
            'deg_gt4_fraction': float(np.mean(deg > 4.0)) if deg.size else 0.0,
            'arterial_length_share': float(l_art / total_len), 'collector_length_share': float(l_col / total_len), 'local_length_share': float(l_loc / total_len),
            'crossing_fraction': float(crossing_fraction), 'main_orientation_modes': float(main_modes), 'off_axis_edge_fraction': float(off_axis),
            'builtup_area_fraction': float(builtup_area_fraction), 'barrier_area_fraction': float(barrier_area_fraction),
        }
        return stats

    def _stats_penalty(self, stats: Dict[str, float]) -> float:
        if not self.target_stats:
            return 0.0
        pen = 0.0
        for key, (lo, hi) in self.target_stats.items():
            val = float(stats.get(str(key), 0.0))
            if val < float(lo):
                pen += float((float(lo) - val) / max(abs(float(hi) - float(lo)), 1e-6))
            elif val > float(hi):
                pen += float((val - float(hi)) / max(abs(float(hi) - float(lo)), 1e-6))
        if int(round(float(stats.get('main_orientation_modes', 0.0)))) != 2:
            pen += 0.35
        return float(max(pen, 0.0))

    def _estimate_field_area_fraction(self, field_fn, threshold: float, samples_per_axis: int = 50) -> float:
        n = int(max(samples_per_axis, 12))
        xs = np.linspace(0.0, float(self.map_size_m), n)
        ys = np.linspace(0.0, float(self.map_size_m), n)
        hit = 0
        total = n * n
        for x in xs:
            for y in ys:
                if float(field_fn(float(x), float(y))) >= float(threshold):
                    hit += 1
        return float(hit / max(total, 1))

    def _compute_crossing_fraction(self, node_xy: Dict[int, Tuple[float, float]], graph: nx.Graph) -> float:
        edges = [(int(u), int(v)) for u, v in graph.edges()]
        if len(edges) <= 1:
            return 0.0
        cross = 0
        for i in range(len(edges) - 1):
            a, b = edges[i]
            for j in range(i + 1, len(edges)):
                c, d = edges[j]
                if len({a, b, c, d}) < 4:
                    continue
                if self._segments_intersect(node_xy[a], node_xy[b], node_xy[c], node_xy[d]):
                    cross += 1
        return float(cross / max(len(edges), 1))

    @staticmethod
    def _segments_intersect(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
        def orient(a, b, c):
            return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        def on_seg(a, b, c):
            return bool(min(a[0], b[0]) - 1e-9 <= c[0] <= max(a[0], b[0]) + 1e-9 and min(a[1], b[1]) - 1e-9 <= c[1] <= max(a[1], b[1]) + 1e-9)
        o1 = orient(p1, p2, p3); o2 = orient(p1, p2, p4); o3 = orient(p3, p4, p1); o4 = orient(p3, p4, p2)
        if (o1 * o2 < 0.0) and (o3 * o4 < 0.0):
            return True
        if abs(o1) <= 1e-9 and on_seg(p1, p2, p3): return True
        if abs(o2) <= 1e-9 and on_seg(p1, p2, p4): return True
        if abs(o3) <= 1e-9 and on_seg(p3, p4, p1): return True
        if abs(o4) <= 1e-9 and on_seg(p3, p4, p2): return True
        return False

    def _edge_crosses_existing(self, node_xy: Dict[int, Tuple[float, float]], graph: nx.Graph, u: int, v: int, ignore_nodes: Optional[Set[int]] = None) -> bool:
        ign = set(ignore_nodes or set())
        for a, b in graph.edges():
            aa, bb = int(a), int(b)
            if len({int(u), int(v), aa, bb}) < 4:
                continue
            if aa in ign or bb in ign:
                continue
            if self._segments_intersect(node_xy[int(u)], node_xy[int(v)], node_xy[aa], node_xy[bb]):
                return True
        return False

    @staticmethod
    def _orientation_bin(x1: float, y1: float, x2: float, y2: float, bins: int = 12) -> int:
        theta = float(np.arctan2(y2 - y1, x2 - x1))
        if theta < 0.0:
            theta += np.pi
        if theta >= np.pi:
            theta -= np.pi
        step = np.pi / float(max(int(bins), 1))
        return int(np.clip(np.floor(theta / max(step, 1e-9)), 0, max(int(bins) - 1, 0)))

    def get_map_stats(self) -> Dict[str, float]:
        return dict(self.map_stats)

    def get_scene_payload(self) -> Dict[str, Any]:
        return dict(self.scene_payload)

    def _cache_distances(self) -> None:
        n = self.node_count
        xy = np.array([self.node_xy[i] for i in range(n)], dtype=np.float64)
        diff = xy[:, None, :] - xy[None, :, :]
        self.euclidean_dist_matrix = np.linalg.norm(diff, axis=2).astype(np.float64)

        sp = np.full((n, n), np.inf, dtype=np.float64)
        for i in range(n):
            sp[i, i] = 0.0
        all_pairs = nx.all_pairs_dijkstra_path_length(self.graph, weight="weight")
        for src, dmap in all_pairs:
            s = int(src)
            for dst, d in dmap.items():
                sp[s, int(dst)] = float(d)
        self.shortest_path_matrix = sp

    def get_euclidean_distance(self, src: int, dst: int) -> float:
        if self.euclidean_dist_matrix is None:
            raise RuntimeError("euclidean_dist_matrix is not initialized")
        return float(self.euclidean_dist_matrix[int(src), int(dst)])

    def get_shortest_path_distance(self, src: int, dst: int) -> float:
        if self.shortest_path_matrix is None:
            raise RuntimeError("shortest_path_matrix is not initialized")
        return float(self.shortest_path_matrix[int(src), int(dst)])

    def get_neighbors(self, node_id: int) -> List[int]:
        return sorted(int(v) for v in self.graph.neighbors(int(node_id)))

    def average_degree(self) -> float:
        n = len(self.node_xy)
        if n <= 0:
            return 0.0
        return float(2.0 * self.graph.number_of_edges() / float(n))

    def to_topology_payload(
        self,
    ) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, Set[int]], np.ndarray, np.ndarray]:
        n = len(self.node_xy)
        adjacency: Dict[int, Set[int]] = {i: set() for i in range(n)}
        for u, v in self.graph.edges():
            adjacency[int(u)].add(int(v))
            adjacency[int(v)].add(int(u))
        if self.euclidean_dist_matrix is None or self.shortest_path_matrix is None:
            raise RuntimeError("distance matrices are not initialized")
        return (
            dict(self.node_xy),
            adjacency,
            self.euclidean_dist_matrix.copy(),
            self.shortest_path_matrix.copy(),
        )

