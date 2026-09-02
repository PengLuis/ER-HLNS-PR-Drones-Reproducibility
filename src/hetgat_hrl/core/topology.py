from __future__ import annotations

import heapq
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from hetgat_hrl.core.disaster_map_graph import DisasterMapGraph
from hetgat_hrl.core.mdp_spec import EnvConfig
from hetgat_hrl.core.real_city_case import build_real_city_case_payload


@dataclass(frozen=True)
class Node:
    node_id: int
    x: float
    y: float
    elevation_m: float = 0.0
    slope_norm: float = 0.0
    node_role: str = "ordinary"
    builtup_intensity: float = 0.0
    barrier_proximity: float = 0.0
    quake_norm: float = 0.0
    pga_g: float = 0.0
    pgv_cms: float = 0.0
    mmi: float = 0.0


@dataclass
class EdgeAttr:
    roughness_norm: float = 0.0
    building_density_norm: float = 0.0
    infra_bottleneck_norm: float = 0.0
    base_vulnerability: float = 0.0
    road_class: str = "collector"
    length_m: float = 0.0
    travel_speed_kph: float = 35.0
    lanes: int = 1
    capacity_class: str = "medium"
    bridge_or_tunnel: bool = False
    barrier_exposure: float = 0.0
    builtup_exposure: float = 0.0
    orientation_bin: int = 0


class GraphTopology:
    """
    Topology layer with two sources:
    1) synthetic random connected graph
    2) OSM + DEM driven graph (fallback to synthetic when unavailable)
    """

    def __init__(
        self,
        nodes: Dict[int, Node],
        adjacency: Dict[int, Set[int]],
        edge_attrs: Optional[Dict[Tuple[int, int], EdgeAttr]] = None,
        euclidean_dist_matrix: Optional[np.ndarray] = None,
        shortest_path_matrix: Optional[np.ndarray] = None,
        real_case_meta: Optional[Dict[str, Any]] = None,
    ):
        self.nodes = nodes
        self.adjacency = {int(k): set(int(x) for x in v) for k, v in adjacency.items()}
        self.blocked_edges: Set[Tuple[int, int]] = set()
        # Dynamic shortest-path caches are keyed by the road-state version
        # and source node.  The version changes only when a blocked edge is
        # actually added/removed, so repeated decisions on the same state
        # reuse one Dijkstra result without returning stale routes.
        self._road_version: int = 0
        self._distance_cache: "OrderedDict[Tuple[int, int], Dict[int, float]]" = OrderedDict()
        self._ignore_blocked_distance_cache: "OrderedDict[int, Dict[int, float]]" = OrderedDict()
        self._distance_cache_limit: int = 64
        self.edge_attrs: Dict[Tuple[int, int], EdgeAttr] = edge_attrs or {}
        self.euclidean_dist_matrix: Optional[np.ndarray] = euclidean_dist_matrix
        self.shortest_path_matrix: Optional[np.ndarray] = shortest_path_matrix
        self._matrix_node_index: Dict[int, int] = self._build_matrix_node_index()
        self.real_case_meta: Dict[str, Any] = dict(real_case_meta or {})
        if not self.edge_attrs:
            self._init_default_edge_attrs()

    def _build_matrix_node_index(self) -> Dict[int, int]:
        """Map matrix rows to node IDs, including sparse/non-contiguous IDs."""
        matrix = self.shortest_path_matrix
        if matrix is None or getattr(matrix, "ndim", 0) != 2:
            return {}
        n = len(self.nodes)
        if matrix.shape[0] != n or matrix.shape[1] != n:
            return {}
        node_ids = sorted(int(node_id) for node_id in self.nodes)
        if node_ids == list(range(n)):
            return {node_id: node_id for node_id in node_ids}
        return {node_id: idx for idx, node_id in enumerate(node_ids)}

    @staticmethod
    def edge_key(src: int, dst: int) -> Tuple[int, int]:
        return (min(int(src), int(dst)), max(int(src), int(dst)))


    @staticmethod
    def _edge_u01(src: int, dst: int, salt: int = 0) -> float:
        """
        Deterministic pseudo-random U(0,1) from edge key.
        Keeps defaults reproducible across runs/platforms.
        """
        a = int(min(src, dst))
        b = int(max(src, dst))
        x = ((a * 73856093) ^ (b * 19349663) ^ (int(salt) * 83492791)) & 0xFFFFFFFF
        x ^= (x >> 13)
        x = (x * 1274126177) & 0xFFFFFFFF
        x ^= (x >> 16)
        return float((x + 0.5) / 4294967296.0)

    def _default_edge_attr(
        self,
        src: int,
        dst: int,
        cut_edges: Optional[Set[Tuple[int, int]]] = None,
    ) -> EdgeAttr:
        a = self.nodes[int(src)]
        b = self.nodes[int(dst)]
        slope = float(np.clip(0.5 * (a.slope_norm + b.slope_norm), 0.0, 1.0))

        # Roughness prior: steeper links are rougher.
        rough = float(np.clip(0.15 + 0.70 * slope, 0.0, 1.0))

        # Building-density default prior (literature-aligned urban morphology bands):
        # low/mid/high coverage ~= 20% / 60% / 20%.
        # Values mapped to normalized density in [0,1].
        # Midpoint closer to map center tends to be denser.
        xs = np.array([n.x for n in self.nodes.values()], dtype=np.float64)
        ys = np.array([n.y for n in self.nodes.values()], dtype=np.float64)
        cx = float(0.5 * (xs.min() + xs.max())) if xs.size else 0.0
        cy = float(0.5 * (ys.min() + ys.max())) if ys.size else 0.0
        diag = float(
            np.hypot(
                max(xs.max() - xs.min(), 1e-6),
                max(ys.max() - ys.min(), 1e-6),
            )
        ) if xs.size else 1.0
        mx = 0.5 * (a.x + b.x)
        my = 0.5 * (a.y + b.y)
        r_norm = float(np.clip(np.hypot(mx - cx, my - cy) / max(0.5 * diag, 1e-6), 0.0, 1.5))
        urbanity = float(np.exp(-((r_norm / 0.95) ** 2)))

        u_b = self._edge_u01(src, dst, salt=1)
        if u_b < 0.20:
            b_base = 0.22
        elif u_b < 0.80:
            b_base = 0.40
        else:
            b_base = 0.58
        bldg = float(np.clip(b_base + 0.12 * (urbanity - 0.5), 0.0, 1.0))

        # Bridge/tunnel/critical-corridor default ratio:
        # tunnel 2%, bridge 6%, other critical 12%, regular 80%.
        # These are structure-type priors, then boosted by graph bottleneck structure.
        u_i = self._edge_u01(src, dst, salt=2)
        if u_i < 0.02:
            infra = 1.00  # tunnel-like critical segment
        elif u_i < 0.08:
            infra = 0.85  # bridge-like critical segment
        elif u_i < 0.20:
            infra = 0.60  # other key corridor
        else:
            infra = 0.20  # regular road segment

        # Structural bottleneck boost from topology.
        deg_src = len(self.adjacency.get(int(src), set()))
        deg_dst = len(self.adjacency.get(int(dst), set()))
        if deg_src <= 2 or deg_dst <= 2:
            infra = max(infra, 0.45)
        if cut_edges is not None and self.edge_key(src, dst) in cut_edges:
            infra = max(infra, 0.60)
        infra = float(np.clip(infra, 0.0, 1.0))

        # V_base = clip((Slope/35)*(1+0.65*D_bldg)*(1+C_infra), 0, 1)
        # slope_norm already corresponds to normalized slope term.
        v_base = float(np.clip((slope) * (1.0 + 0.65 * bldg) * (1.0 + infra), 0.0, 1.0))
        return EdgeAttr(
            roughness_norm=rough,
            building_density_norm=bldg,
            infra_bottleneck_norm=infra,
            base_vulnerability=v_base,
            road_class="collector",
            length_m=float(np.hypot(a.x - b.x, a.y - b.y)),
            travel_speed_kph=35.0,
            lanes=1,
            capacity_class="medium",
            bridge_or_tunnel=False,
            barrier_exposure=float(np.clip(slope, 0.0, 1.0)),
            builtup_exposure=float(np.clip(bldg, 0.0, 1.0)),
            orientation_bin=int(round(((np.degrees(np.arctan2(b.y - a.y, b.x - a.x)) + 360.0) % 180.0) / 22.5)) % 8,
        )

    def _init_default_edge_attrs(self) -> None:
        cut_edges = self._bridge_edges(len(self.nodes), self.adjacency)
        for src, nbs in self.adjacency.items():
            for dst in nbs:
                if src >= dst:
                    continue
                self.edge_attrs[(src, dst)] = self._default_edge_attr(
                    src=src, dst=dst, cut_edges=cut_edges
                )

    @staticmethod
    def _mean_degree(adjacency: Dict[int, Set[int]]) -> float:
        n = max(len(adjacency), 1)
        m2 = float(sum(len(v) for v in adjacency.values()))
        return m2 / float(n)

    def average_degree(self) -> float:
        return self._mean_degree(self.adjacency)

    @staticmethod
    def _target_edge_count(
        num_nodes: int, num_edges: int, avg_degree_min: float, avg_degree_max: float
    ) -> int:
        min_edges = int(np.ceil(0.5 * avg_degree_min * num_nodes))
        max_edges = int(np.floor(0.5 * avg_degree_max * num_nodes))
        if min_edges > max_edges:
            min_edges = max_edges = max(1, int(round(0.5 * 3.5 * num_nodes)))
        if num_edges <= 0:
            return int(round(0.5 * 0.5 * (avg_degree_min + avg_degree_max) * num_nodes))
        return int(np.clip(num_edges, min_edges, max_edges))

    @staticmethod
    def _as_undirected_edges(adjacency: Dict[int, Set[int]]) -> Set[Tuple[int, int]]:
        es: Set[Tuple[int, int]] = set()
        for a, nbs in adjacency.items():
            for b in nbs:
                if a == b:
                    continue
                es.add((min(a, b), max(a, b)))
        return es

    @staticmethod
    def _rebuild_adjacency(num_nodes: int, edges: Set[Tuple[int, int]]) -> Dict[int, Set[int]]:
        out: Dict[int, Set[int]] = {i: set() for i in range(num_nodes)}
        for a, b in edges:
            out[a].add(b)
            out[b].add(a)
        return out

    @staticmethod
    def _is_connected(num_nodes: int, adjacency: Dict[int, Set[int]]) -> bool:
        if num_nodes <= 1:
            return True
        seen = set()
        stack = [0]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for nb in adjacency.get(cur, set()):
                if nb not in seen:
                    stack.append(nb)
        return len(seen) == num_nodes

    @staticmethod
    def _connected_components(adjacency: Dict[int, Set[int]]) -> List[Set[int]]:
        nodes = set(adjacency.keys())
        comps: List[Set[int]] = []
        while nodes:
            root = nodes.pop()
            comp = {root}
            stack = [root]
            while stack:
                cur = stack.pop()
                for nb in adjacency.get(cur, set()):
                    if nb not in comp:
                        comp.add(nb)
                        if nb in nodes:
                            nodes.remove(nb)
                        stack.append(nb)
            comps.append(comp)
        return comps

    @staticmethod
    def _bridge_edges(num_nodes: int, adjacency: Dict[int, Set[int]]) -> Set[Tuple[int, int]]:
        bridges: Set[Tuple[int, int]] = set()
        base_edges = GraphTopology._as_undirected_edges(adjacency)
        for e in list(base_edges):
            a, b = e
            adj2 = {k: set(v) for k, v in adjacency.items()}
            adj2[a].discard(b)
            adj2[b].discard(a)
            if not GraphTopology._is_connected(num_nodes, adj2):
                bridges.add(e)
        return bridges

    @staticmethod
    def _ensure_connected(
        nodes: Dict[int, Node], adjacency: Dict[int, Set[int]]
    ) -> Dict[int, Set[int]]:
        out = {k: set(v) for k, v in adjacency.items()}
        num_nodes = len(nodes)
        if num_nodes <= 1:
            return out
        comps = GraphTopology._connected_components(out)
        while len(comps) > 1:
            c1 = comps[0]
            c2 = comps[1]
            best = None
            best_d = float("inf")
            for a in c1:
                for b in c2:
                    na, nb = nodes[a], nodes[b]
                    d = float(np.hypot(na.x - nb.x, na.y - nb.y))
                    if d < best_d:
                        best_d = d
                        best = (a, b)
            if best is None:
                break
            a, b = best
            out[a].add(b)
            out[b].add(a)
            comps = GraphTopology._connected_components(out)
        return out

    @staticmethod
    def _rebalance_degree_band(
        nodes: Dict[int, Node],
        adjacency: Dict[int, Set[int]],
        target_edges: int,
    ) -> Dict[int, Set[int]]:
        out = GraphTopology._ensure_connected(nodes, adjacency)
        num_nodes = len(nodes)
        edges = GraphTopology._as_undirected_edges(out)

        # Add edges if sparse.
        while len(edges) < target_edges:
            best = None
            best_d = float("inf")
            for a in range(num_nodes):
                for b in range(a + 1, num_nodes):
                    if (a, b) in edges:
                        continue
                    na, nb = nodes[a], nodes[b]
                    d = float(np.hypot(na.x - nb.x, na.y - nb.y))
                    if d < best_d:
                        best_d = d
                        best = (a, b)
            if best is None:
                break
            edges.add(best)

        # Remove edges if dense while preserving connectivity.
        while len(edges) > target_edges:
            adj_tmp = GraphTopology._rebuild_adjacency(num_nodes, edges)
            bridges = GraphTopology._bridge_edges(num_nodes, adj_tmp)
            removable = [e for e in edges if e not in bridges]
            if not removable:
                break
            # Prefer removing long non-bridge edges.
            removable.sort(
                key=lambda e: np.hypot(
                    nodes[e[0]].x - nodes[e[1]].x, nodes[e[0]].y - nodes[e[1]].y
                ),
                reverse=True,
            )
            edges.remove(removable[0])

        return GraphTopology._rebuild_adjacency(num_nodes, edges)

    @staticmethod
    def _normalize_xy(
        raw_xy: Dict[int, Tuple[float, float]], world_size_m: float
    ) -> Dict[int, Tuple[float, float]]:
        xs = np.array([xy[0] for xy in raw_xy.values()], dtype=np.float64)
        ys = np.array([xy[1] for xy in raw_xy.values()], dtype=np.float64)
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        dx = max(x1 - x0, 1e-6)
        dy = max(y1 - y0, 1e-6)
        out: Dict[int, Tuple[float, float]] = {}
        for k, (x, y) in raw_xy.items():
            xn = (x - x0) / dx
            yn = (y - y0) / dy
            out[k] = (float(xn * world_size_m), float(yn * world_size_m))
        return out

    @staticmethod
    def _load_dem(dem_npy_path: str) -> Optional[np.ndarray]:
        p = Path(str(dem_npy_path)) if dem_npy_path else None
        if p is None or not p.exists():
            return None
        if p.suffix.lower() != ".npy":
            return None
        try:
            arr = np.load(str(p))
            if arr.ndim != 2:
                return None
            return arr.astype(np.float64)
        except Exception:
            return None

    @staticmethod
    def _annotate_dem(
        nodes_xy: Dict[int, Tuple[float, float]],
        world_size_m: float,
        dem: Optional[np.ndarray],
    ) -> Dict[int, Tuple[float, float]]:
        if dem is None:
            return {k: (0.0, 0.0) for k in nodes_xy}
        gy, gx = np.gradient(dem)
        slope_field = np.hypot(gx, gy)
        s_ref = float(np.percentile(slope_field, 95))
        s_ref = max(s_ref, 1e-6)
        h, w = dem.shape
        out: Dict[int, Tuple[float, float]] = {}
        for k, (x, y) in nodes_xy.items():
            u = float(np.clip(x / max(world_size_m, 1e-6), 0.0, 1.0))
            v = float(np.clip(y / max(world_size_m, 1e-6), 0.0, 1.0))
            j = int(round(u * (w - 1)))
            i = int(round(v * (h - 1)))
            elev = float(dem[i, j])
            slope = float(np.clip(slope_field[i, j] / s_ref, 0.0, 1.0))
            out[k] = (elev, slope)
        return out

    @staticmethod
    def _safe_float(d: dict, keys: List[str]) -> Optional[float]:
        for k in keys:
            if k in d:
                try:
                    return float(d[k])
                except Exception:
                    continue
        return None

    @staticmethod
    def generate_random_connected(
        num_nodes: int,
        num_edges: int,
        seed: int,
        world_size_m: float = 3000.0,
        avg_degree_min: float = 3.2,
        avg_degree_max: float = 3.8,
    ) -> "GraphTopology":
        rng = np.random.default_rng(seed)
        coords = rng.uniform(0.0, world_size_m, size=(num_nodes, 2))
        nodes = {
            i: Node(node_id=i, x=float(coords[i, 0]), y=float(coords[i, 1]))
            for i in range(num_nodes)
        }
        adjacency: Dict[int, Set[int]] = {i: set() for i in range(num_nodes)}

        # Local-first backbone: kNN candidate graph + nearest-component bridges.
        if num_nodes > 1:
            dmat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
            local_k = int(np.clip(int(round(0.5 * (avg_degree_min + avg_degree_max))) + 1, 3, 6))
            cand: Dict[int, Set[int]] = {i: set() for i in range(num_nodes)}

            for i in range(num_nodes):
                order = np.argsort(dmat[i])
                for j in order[1 : local_k + 1]:
                    jj = int(j)
                    cand[i].add(jj)
                    cand[jj].add(i)

            cand = GraphTopology._ensure_connected(nodes, cand)
            cand_edges = GraphTopology._as_undirected_edges(cand)
            if cand_edges:
                import networkx as nx

                g = nx.Graph()
                g.add_nodes_from(range(num_nodes))
                for a, b in cand_edges:
                    g.add_edge(int(a), int(b), weight=float(dmat[int(a), int(b)]))
                mst = nx.minimum_spanning_tree(g, algorithm="kruskal", weight="weight")
                for a, b in mst.edges():
                    aa = int(a)
                    bb = int(b)
                    adjacency[aa].add(bb)
                    adjacency[bb].add(aa)
            else:
                # Fallback for degenerate tiny graphs.
                order = list(rng.permutation(num_nodes))
                for i in range(num_nodes - 1):
                    a, b = int(order[i]), int(order[i + 1])
                    adjacency[a].add(b)
                    adjacency[b].add(a)

        target_edges = GraphTopology._target_edge_count(
            num_nodes, num_edges, avg_degree_min, avg_degree_max
        )
        adjacency = GraphTopology._rebalance_degree_band(nodes, adjacency, target_edges)
        return GraphTopology(nodes=nodes, adjacency=adjacency)

    @staticmethod
    @staticmethod
    def generate_from_osm_dem(
        num_nodes: int,
        num_edges: int,
        seed: int,
        osm_graphml_path: str,
        dem_npy_path: str,
        world_size_m: float = 3000.0,
        avg_degree_min: float = 3.2,
        avg_degree_max: float = 3.8,
    ) -> "GraphTopology":
        try:
            import networkx as nx  # noqa: F401
        except Exception:
            return GraphTopology.generate_random_connected(
                num_nodes=num_nodes,
                num_edges=num_edges,
                seed=seed,
                world_size_m=world_size_m,
                avg_degree_min=avg_degree_min,
                avg_degree_max=avg_degree_max,
            )

        p = Path(str(osm_graphml_path)) if osm_graphml_path else None
        if p is None or not p.exists():
            return GraphTopology.generate_random_connected(
                num_nodes=num_nodes,
                num_edges=num_edges,
                seed=seed,
                world_size_m=world_size_m,
                avg_degree_min=avg_degree_min,
                avg_degree_max=avg_degree_max,
            )

        payload = build_real_city_case_payload(
            EnvConfig(
                seed=int(seed),
                map_source="osm_dem",
                osm_graphml_path=str(osm_graphml_path),
                dem_npy_path=str(dem_npy_path),
                map_size_m=float(world_size_m),
                real_case_enabled=True,
                real_city_case="generic_osm_dem",
                real_case_name="generic_osm_dem",
                real_case_size_m=float(world_size_m),
            )
        )
        nodes = {
            int(i): Node(
                node_id=int(i),
                x=float(data["x"]),
                y=float(data["y"]),
                elevation_m=float(data.get("elevation_m", 0.0)),
                slope_norm=float(data.get("slope_norm", 0.0)),
                node_role=str(data.get("node_role", "ordinary")),
                builtup_intensity=float(data.get("builtup_intensity", 0.0)),
                barrier_proximity=float(data.get("barrier_proximity", 0.0)),
            )
            for i, data in payload["nodes_raw"].items()
        }
        edge_attrs = {
            (int(k[0]), int(k[1])): EdgeAttr(**dict(v))
            for k, v in payload["edge_attrs"].items()
        }
        return GraphTopology(
            nodes=nodes,
            adjacency=payload["adjacency"],
            edge_attrs=edge_attrs,
            real_case_meta=payload.get("real_case_meta", {}),
        )

    @staticmethod
    def generate_from_disaster_map(cfg: EnvConfig) -> "GraphTopology":
        """
        Build topology from DisasterMapGraph with explicit config controls.
        """
        # Always honor explicit config values from paper pipeline so
        # train/eval/viewer use the same topology controls.
        target_stats = {
            "num_nodes": (float(cfg.l_target_num_nodes_min), float(cfg.l_target_num_nodes_max)),
            "num_edges": (float(cfg.l_target_num_edges_min), float(cfg.l_target_num_edges_max)),
            "avg_degree": (float(cfg.l_target_avg_degree_min), float(cfg.l_target_avg_degree_max)),
            "median_edge_length_m": (float(cfg.l_target_median_edge_length_m_min), float(cfg.l_target_median_edge_length_m_max)),
            "p90_edge_length_m": (float(cfg.l_target_p90_edge_length_m_min), float(cfg.l_target_p90_edge_length_m_max)),
            "leaf_fraction": (0.0, float(cfg.l_target_leaf_fraction_max)),
            "deg3_fraction": (float(cfg.l_target_deg3_fraction_min), float(cfg.l_target_deg3_fraction_max)),
            "deg4_fraction": (float(cfg.l_target_deg4_fraction_min), float(cfg.l_target_deg4_fraction_max)),
            "deg_gt4_fraction": (0.0, float(cfg.l_target_deg_gt4_fraction_max)),
            "arterial_length_share": (float(cfg.l_target_arterial_length_share_min), float(cfg.l_target_arterial_length_share_max)),
            "collector_length_share": (float(cfg.l_target_collector_length_share_min), float(cfg.l_target_collector_length_share_max)),
            "local_length_share": (float(cfg.l_target_local_length_share_min), float(cfg.l_target_local_length_share_max)),
            "crossing_fraction": (0.0, float(cfg.l_target_max_crossing_fraction)),
            "off_axis_edge_fraction": (float(cfg.l_target_off_axis_edge_fraction_min), float(cfg.l_target_off_axis_edge_fraction_max)),
            "builtup_area_fraction": (float(cfg.l_target_builtup_area_fraction_min), float(cfg.l_target_builtup_area_fraction_max)),
            "barrier_area_fraction": (float(cfg.l_target_barrier_area_fraction_min), float(cfg.l_target_barrier_area_fraction_max)),
        }
        cache_idx = int(getattr(cfg, "l_map_cache_index", -1))
        gen = DisasterMapGraph(
            seed=int(cfg.seed),
            map_size_m=float(cfg.map_size_m),
            node_count=int(cfg.num_nodes or cfg.n_nodes),
            min_node_spacing_m=float(cfg.min_node_spacing_m),
            redundant_edge_radius_m=float(cfg.redundant_edge_radius_m),
            redundant_edge_prob=float(cfg.redundant_edge_prob),
            target_avg_degree=float(0.5 * (float(cfg.avg_degree_min) + float(cfg.avg_degree_max))),
            max_degree=int(max(3, int(np.ceil(float(cfg.avg_degree_max))) + 1)),
            map_complexity=str(getattr(cfg, "map_complexity", "") or getattr(cfg, "phase", "")).upper(),
            quality_gate_max_attempts=int(max(1, int(getattr(cfg, "l_map_generation_max_attempts", 1)))),
            target_stats=target_stats,
            l_map_variant=str(getattr(cfg, "l_map_variant", "L_v1b_orientation_tighter")),
            l_map_acceptance_mode=str(getattr(cfg, "l_map_acceptance_mode", "realism_first")),
            l_min_node_spacing_m=float(getattr(cfg, "l_min_node_spacing_m", 220.0)),
            l_min_gateway_spacing_m=float(getattr(cfg, "l_min_gateway_spacing_m", 300.0)),
            l_min_arterial_junction_spacing_m=float(getattr(cfg, "l_min_arterial_junction_spacing_m", 350.0)),
            l_collinear_triangle_angle_deg=float(getattr(cfg, "l_collinear_triangle_angle_deg", 165.0)),
            task_normal_count=int(getattr(cfg, "num_normal_tasks", 8)),
            task_emergency_count=int(getattr(cfg, "num_emergency_tasks", 12)),
            task_min_spacing_m=float(getattr(cfg, "l_task_min_spacing_m", 240.0)),
            use_map_cache=bool(getattr(cfg, "l_map_cache_enabled", False)),
            map_cache_dir=str(getattr(cfg, "l_map_cache_dir", "data/maps_core/L_map")),
            map_cache_size=int(max(1, int(getattr(cfg, "l_map_cache_size", 20)))),
            map_cache_index=(None if cache_idx < 0 else cache_idx),
        )
        node_xy, adjacency, euclid_m, sp_m = gen.to_topology_payload()
        nodes = {
            int(i): Node(node_id=int(i), x=float(x), y=float(y))
            for i, (x, y) in node_xy.items()
        }
        for nid, attrs in getattr(gen, "node_attrs", {}).items():
            if int(nid) not in nodes:
                continue
            old = nodes[int(nid)]
            nodes[int(nid)] = Node(
                node_id=int(old.node_id),
                x=float(old.x),
                y=float(old.y),
                node_role=str(attrs.get("node_role", old.node_role)),
                builtup_intensity=float(attrs.get("builtup_intensity", old.builtup_intensity)),
                barrier_proximity=float(attrs.get("barrier_proximity", old.barrier_proximity)),
            )
        edge_attrs: Dict[Tuple[int, int], EdgeAttr] = {}
        for u, v, data in gen.graph.edges(data=True):
            key = GraphTopology.edge_key(int(u), int(v))
            edge_attrs[key] = EdgeAttr(
                road_class=str(data.get("road_class", "collector")),
                length_m=float(data.get("length_m", data.get("weight", 0.0))),
                travel_speed_kph=float(data.get("travel_speed_kph", 35.0)),
                lanes=int(data.get("lanes", 1)),
                capacity_class=str(data.get("capacity_class", "medium")),
                bridge_or_tunnel=bool(data.get("bridge_or_tunnel", False)),
                barrier_exposure=float(data.get("barrier_exposure", 0.0)),
                building_density_norm=float(data.get("builtup_exposure", 0.0)),
                builtup_exposure=float(data.get("builtup_exposure", 0.0)),
                orientation_bin=int(data.get("orientation_bin", 0)),
            )
        scene_payload = gen.get_scene_payload()
        synthetic_meta: Dict[str, Any] = {
            "synthetic_realism": True,
            "scene_payload": dict(scene_payload),
            "map_stats": gen.get_map_stats(),
            "major_clusters": list(scene_payload.get("major_clusters", [])) if isinstance(scene_payload, dict) else [],
            "cluster_id_by_node": {},
            "synthetic_realism_tasks": dict(scene_payload.get("tasks", {})) if isinstance(scene_payload, dict) else {},
        }
        for cid, comp in enumerate(synthetic_meta.get("major_clusters", [])):
            for nid in comp:
                synthetic_meta["cluster_id_by_node"][int(nid)] = int(cid)
        return GraphTopology(
            nodes=nodes,
            adjacency=adjacency,
            edge_attrs=edge_attrs,
            euclidean_dist_matrix=euclid_m,
            shortest_path_matrix=sp_m,
            real_case_meta=synthetic_meta,
        )

    @staticmethod
    def build_from_config(cfg: EnvConfig) -> "GraphTopology":
        source = str(cfg.map_source).strip().lower()
        if source in {"disaster_map", "disaster", "m_complexity"}:
            return GraphTopology.generate_from_disaster_map(cfg)
        if source == "osm_dem":
            if bool(getattr(cfg, "real_case_enabled", False)):
                payload = build_real_city_case_payload(cfg)
                nodes = {
                    int(i): Node(
                        node_id=int(i),
                        x=float(data["x"]),
                        y=float(data["y"]),
                        elevation_m=float(data.get("elevation_m", 0.0)),
                        slope_norm=float(data.get("slope_norm", 0.0)),
                        node_role=str(data.get("node_role", "ordinary")),
                        builtup_intensity=float(data.get("builtup_intensity", 0.0)),
                        barrier_proximity=float(data.get("barrier_proximity", 0.0)),
                        quake_norm=float(data.get("quake_norm", 0.0)),
                        pga_g=float(data.get("pga_g", 0.0)),
                        pgv_cms=float(data.get("pgv_cms", 0.0)),
                        mmi=float(data.get("mmi", 0.0)),
                    )
                    for i, data in payload["nodes_raw"].items()
                }
                edge_attrs = {
                    (int(k[0]), int(k[1])): EdgeAttr(**dict(v))
                    for k, v in payload["edge_attrs"].items()
                }
                return GraphTopology(
                    nodes=nodes,
                    adjacency=payload["adjacency"],
                    edge_attrs=edge_attrs,
                    real_case_meta=payload.get("real_case_meta", {}),
                )
            return GraphTopology.generate_from_osm_dem(
                num_nodes=int(cfg.num_nodes or cfg.n_nodes),
                num_edges=int(cfg.num_edges),
                seed=int(cfg.seed),
                osm_graphml_path=str(cfg.osm_graphml_path or ""),
                dem_npy_path=str(cfg.dem_npy_path or ""),
                world_size_m=float(cfg.map_size_m),
                avg_degree_min=float(cfg.avg_degree_min),
                avg_degree_max=float(cfg.avg_degree_max),
            )
        return GraphTopology.generate_random_connected(
            num_nodes=int(cfg.num_nodes or cfg.n_nodes),
            num_edges=int(cfg.num_edges),
            seed=int(cfg.seed),
            world_size_m=float(cfg.map_size_m),
            avg_degree_min=float(cfg.avg_degree_min),
            avg_degree_max=float(cfg.avg_degree_max),
        )

    def neighbors(self, node_id: int) -> List[int]:
        nbs = []
        for nb in self.adjacency.get(node_id, set()):
            if not self.is_blocked(node_id, nb):
                nbs.append(nb)
        return nbs

    def is_connected(self, src: int, dst: int) -> bool:
        return dst in self.adjacency.get(src, set())

    def edge_distance(self, src: int, dst: int) -> float:
        k = self.edge_key(src, dst)
        if k in self.edge_attrs and float(getattr(self.edge_attrs[k], "length_m", 0.0)) > 0.0:
            return float(self.edge_attrs[k].length_m)
        if self.euclidean_dist_matrix is not None:
            return float(self.euclidean_dist_matrix[int(src), int(dst)])
        a = self.nodes[src]
        b = self.nodes[dst]
        return float(np.hypot(a.x - b.x, a.y - b.y))

    def edge_attr(self, src: int, dst: int) -> EdgeAttr:
        k = self.edge_key(src, dst)
        if k not in self.edge_attrs:
            self.edge_attrs[k] = self._default_edge_attr(src=int(k[0]), dst=int(k[1]))
        return self.edge_attrs[k]

    def is_blocked(self, src: int, dst: int) -> bool:
        k = (min(src, dst), max(src, dst))
        return k in self.blocked_edges

    def set_blocked(self, src: int, dst: int, blocked: bool) -> None:
        k = self.edge_key(src, dst)
        changed = (k not in self.blocked_edges) if blocked else (k in self.blocked_edges)
        if not changed:
            return
        if blocked:
            self.blocked_edges.add(k)
        else:
            self.blocked_edges.remove(k)
        self._road_version += 1
        # A changed blocked-edge set invalidates all stateful shortest-path
        # results.  Distances computed with blocked edges ignored are purely
        # topological and remain valid across road-state changes.
        self._distance_cache.clear()

    def blocked_ratio(self) -> float:
        total = sum(len(v) for v in self.adjacency.values()) // 2
        if total <= 0:
            return 0.0
        return float(len(self.blocked_edges) / total)

    def path_exists(self, src: int, dst: int, ignore_blocked: bool = False) -> bool:
        if int(src) == int(dst):
            return True
        seen = set()
        stack = [int(src)]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for nb in self.adjacency.get(cur, set()):
                if (not ignore_blocked) and self.is_blocked(cur, nb):
                    continue
                if nb == int(dst):
                    return True
                if nb not in seen:
                    stack.append(nb)
        return False

    def shortest_path_distance(
        self, src: int, dst: int, ignore_blocked: bool = False
    ) -> float:
        src = int(src)
        dst = int(dst)
        if src == dst:
            return 0.0
        # O(1) cache path length when blocked edges are ignored
        # or when the graph currently has no blocked edges.
        if self.shortest_path_matrix is not None and (
            ignore_blocked or len(self.blocked_edges) == 0
        ):
            src_idx = self._matrix_node_index.get(src)
            dst_idx = self._matrix_node_index.get(dst)
            if src_idx is not None and dst_idx is not None:
                return float(self.shortest_path_matrix[src_idx, dst_idx])

        if ignore_blocked:
            cache = self._ignore_blocked_distance_cache
            cache_key = src
        else:
            cache = self._distance_cache
            cache_key = (self._road_version, src)
        distances = cache.get(cache_key)
        if distances is None:
            distances = self._dijkstra_all_distances(src, ignore_blocked=ignore_blocked)
            cache[cache_key] = distances
            cache.move_to_end(cache_key)
            while len(cache) > self._distance_cache_limit:
                cache.popitem(last=False)
        else:
            cache.move_to_end(cache_key)
        return float(distances.get(dst, float("inf")))

    def _dijkstra_all_distances(self, src: int, ignore_blocked: bool = False) -> Dict[int, float]:
        """Compute shortest distances from ``src`` to every node once.

        The graph uses non-contiguous integer IDs in some real-city assets;
        dictionaries and heap entries therefore deliberately carry IDs rather
        than assuming a dense ``0..N-1`` index.
        """
        inf = float("inf")
        distances: Dict[int, float] = {int(node_id): inf for node_id in self.nodes}
        if src not in distances:
            return distances
        distances[src] = 0.0
        heap: List[Tuple[float, int]] = [(0.0, src)]
        while heap:
            cur_d, cur = heapq.heappop(heap)
            if cur_d != distances.get(cur, inf):
                continue
            for nb in self.adjacency.get(cur, set()):
                if not ignore_blocked and self.is_blocked(cur, nb):
                    continue
                nd = cur_d + self.edge_distance(cur, nb)
                if nd < distances.get(nb, inf):
                    distances[nb] = nd
                    heapq.heappush(heap, (nd, int(nb)))
        return distances




