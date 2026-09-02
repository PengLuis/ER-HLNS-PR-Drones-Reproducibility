from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from hetgat_hrl.core.mdp_spec import EnvConfig

_ALLOWED_HIGHWAY = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "residential",
    "unclassified",
}


def _safe_float(data: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        if key in data:
            try:
                return float(data[key])
            except Exception:
                continue
    return None


def _as_listish(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip().lower() for v in value if str(v).strip()]
    txt = str(value).strip()
    if not txt:
        return []
    if txt.startswith("[") and txt.endswith("]"):
        try:
            parsed = ast.literal_eval(txt)
            if isinstance(parsed, (list, tuple, set)):
                return [str(v).strip().lower() for v in parsed if str(v).strip()]
        except Exception:
            pass
    return [txt.lower()]


def _pick_highway(data: Dict[str, Any]) -> str:
    vals = _as_listish(data.get("highway"))
    if not vals:
        return "unclassified"
    priority = {
        "motorway": 0,
        "trunk": 1,
        "primary": 2,
        "secondary": 3,
        "tertiary": 4,
        "residential": 5,
        "unclassified": 6,
        "service": 7,
    }
    vals.sort(key=lambda x: priority.get(x, 99))
    return str(vals[0])


def _road_class_from_highway(highway: str) -> str:
    hw = str(highway).strip().lower()
    if hw in {"motorway", "trunk", "primary"}:
        return "arterial"
    if hw in {"secondary", "tertiary"}:
        return "collector"
    return "local"


def _default_speed_kph(road_class: str) -> float:
    rc = str(road_class).strip().lower()
    if rc == "arterial":
        return 55.0
    if rc == "collector":
        return 38.0
    return 25.0


def _capacity_class(road_class: str, lanes: int) -> str:
    rc = str(road_class).strip().lower()
    if rc == "arterial":
        return "high" if int(lanes) >= 3 else "medium"
    if rc == "collector":
        return "medium"
    return "low"


def _parse_lanes(value: Any, road_class: str) -> int:
    if value is None:
        return 2 if road_class != "local" else 1
    vals = _as_listish(value)
    for v in vals:
        try:
            return max(1, int(round(float(v))))
        except Exception:
            continue
    return 2 if road_class != "local" else 1


def _parse_speed(value: Any, road_class: str) -> float:
    vals = _as_listish(value)
    for v in vals:
        txt = str(v).lower().replace("km/h", "").replace("kph", "").replace("mph", "")
        try:
            raw = float(txt)
            if "mph" in str(v).lower():
                raw *= 1.60934
            return float(max(raw, 5.0))
        except Exception:
            continue
    return _default_speed_kph(road_class)


def _is_bridge_or_tunnel(data: Dict[str, Any]) -> bool:
    bridge = str(data.get("bridge", "")).strip().lower()
    tunnel = str(data.get("tunnel", "")).strip().lower()
    return bridge not in {"", "no", "false", "0", "none"} or tunnel not in {"", "no", "false", "0", "none"}


def _looks_geographic(xy: List[Tuple[float, float]]) -> bool:
    if not xy:
        return False
    xs = [x for x, _ in xy]
    ys = [y for _, y in xy]
    return max(abs(min(xs)), abs(max(xs))) <= 180.0 and max(abs(min(ys)), abs(max(ys))) <= 90.0


def _lonlat_to_local_xy(lon: float, lat: float, center_lon: float, center_lat: float, size_m: float) -> Tuple[float, float]:
    lat_scale = 111320.0
    lon_scale = 111320.0 * math.cos(math.radians(center_lat))
    x = (float(lon) - float(center_lon)) * lon_scale + 0.5 * float(size_m)
    y = (float(lat) - float(center_lat)) * lat_scale + 0.5 * float(size_m)
    return float(x), float(y)


def _orientation_bin(ax: float, ay: float, bx: float, by: float) -> int:
    ang = float((math.degrees(math.atan2(by - ay, bx - ax)) + 360.0) % 180.0)
    return int(round(ang / 22.5)) % 8


def _edge_length_from_nodes(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(math.hypot(x2 - x1, y2 - y1))


def _best_parallel_edge(old: Optional[Dict[str, Any]], new: Dict[str, Any]) -> Dict[str, Any]:
    if old is None:
        return new
    rank = {"arterial": 0, "collector": 1, "local": 2}
    old_rank = rank.get(str(old.get("road_class", "collector")), 9)
    new_rank = rank.get(str(new.get("road_class", "collector")), 9)
    if new_rank < old_rank:
        return new
    if new_rank > old_rank:
        return old
    if float(new.get("length_m", 1e18)) < float(old.get("length_m", 1e18)):
        return new
    return old


def _line_angle_deg(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    bax = a[0] - b[0]
    bay = a[1] - b[1]
    bcx = c[0] - b[0]
    bcy = c[1] - b[1]
    na = math.hypot(bax, bay)
    nc = math.hypot(bcx, bcy)
    if na <= 1e-6 or nc <= 1e-6:
        return 0.0
    dot = max(-1.0, min(1.0, (bax * bcx + bay * bcy) / (na * nc)))
    return float(math.degrees(math.acos(dot)))


def _collapse_degree2_geometry(g: nx.Graph, node_xy: Dict[Any, Tuple[float, float]], angle_thresh_deg: float = 165.0) -> nx.Graph:
    graph = g.copy()
    changed = True
    while changed:
        changed = False
        for n in list(graph.nodes()):
            if n not in graph:
                continue
            if graph.degree(n) != 2:
                continue
            nbs = list(graph.neighbors(n))
            if len(nbs) != 2 or nbs[0] == nbs[1]:
                continue
            a, b = nbs
            e1 = dict(graph.edges[n, a])
            e2 = dict(graph.edges[n, b])
            if bool(e1.get("bridge_or_tunnel", False) or e2.get("bridge_or_tunnel", False)):
                continue
            cls1 = str(e1.get("road_class", "collector"))
            cls2 = str(e2.get("road_class", "collector"))
            ang = _line_angle_deg(node_xy[a], node_xy[n], node_xy[b])
            shortish = float(e1.get("length_m", 0.0)) + float(e2.get("length_m", 0.0)) <= 260.0
            if cls1 != cls2 and ang < angle_thresh_deg and not shortish:
                continue
            if ang < angle_thresh_deg and not shortish:
                continue
            merged = {
                "road_class": cls1 if cls1 == cls2 else (cls1 if cls1 == "arterial" or cls2 == "local" else cls2),
                "length_m": float(e1.get("length_m", 0.0)) + float(e2.get("length_m", 0.0)),
                "travel_speed_kph": float(min(float(e1.get("travel_speed_kph", 35.0)), float(e2.get("travel_speed_kph", 35.0)))),
                "lanes": int(max(int(e1.get("lanes", 1)), int(e2.get("lanes", 1)))),
                "capacity_class": str(e1.get("capacity_class", e2.get("capacity_class", "medium"))),
                "bridge_or_tunnel": False,
                "orientation_bin": int(_orientation_bin(*node_xy[a], *node_xy[b])),
            }
            graph.add_edge(a, b, **_best_parallel_edge(dict(graph.edges[a, b]) if graph.has_edge(a, b) else None, merged))
            graph.remove_node(n)
            changed = True
            break
    return graph


def _compute_pairwise_density(coords: Dict[Any, Tuple[float, float]]) -> Dict[Any, float]:
    keys = list(coords.keys())
    if not keys:
        return {}
    pts = np.asarray([coords[k] for k in keys], dtype=np.float64)
    out: Dict[Any, float] = {}
    for i, k in enumerate(keys):
        d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
        c1 = float(np.sum(d <= 350.0) - 1.0)
        c2 = float(np.sum(d <= 750.0) - 1.0)
        out[k] = float(0.65 * c1 + 0.35 * c2)
    vals = np.asarray(list(out.values()), dtype=np.float64)
    ref = float(np.percentile(vals, 95)) if vals.size else 1.0
    ref = max(ref, 1e-6)
    return {k: float(np.clip(v / ref, 0.0, 1.0)) for k, v in out.items()}


def _compute_barrier_proxy(coords: Dict[Any, Tuple[float, float]], slope_norm: Dict[Any, float], size_m: float) -> Dict[Any, float]:
    out: Dict[Any, float] = {}
    for k, (x, y) in coords.items():
        xn = float(np.clip(float(x) / max(float(size_m), 1e-6), 0.0, 1.0))
        yn = float(np.clip(float(y) / max(float(size_m), 1e-6), 0.0, 1.0))
        nw = float(np.clip(0.65 * (1.0 - xn) + 0.35 * yn, 0.0, 1.0))
        out[k] = float(np.clip(0.55 * nw + 0.45 * float(slope_norm.get(k, 0.0)), 0.0, 1.0))
    return out


def _compute_node_roles(graph: nx.Graph, builtup: Dict[Any, float], barrier: Dict[Any, float]) -> Dict[Any, str]:
    roles: Dict[Any, str] = {}
    for n in graph.nodes():
        deg = int(graph.degree(n))
        cls = [str(graph.edges[n, nb].get("road_class", "collector")) for nb in graph.neighbors(n)]
        arterial = sum(1 for c in cls if c == "arterial")
        collector = sum(1 for c in cls if c == "collector")
        bridge = any(bool(graph.edges[n, nb].get("bridge_or_tunnel", False)) for nb in graph.neighbors(n))
        if bridge:
            roles[n] = "bottleneck"
        elif deg >= 3 and arterial >= 2:
            roles[n] = "arterial_junction"
        elif deg >= 3 and arterial + collector >= 2:
            roles[n] = "collector_junction"
        elif builtup.get(n, 0.0) >= 0.45 and arterial + collector >= 1:
            roles[n] = "area_gateway"
        else:
            roles[n] = "ordinary"
        if barrier.get(n, 0.0) >= 0.72 and roles[n] == "ordinary":
            roles[n] = "bottleneck"
    return roles


def _detect_major_clusters(graph: nx.Graph, builtup: Dict[Any, float], roles: Dict[Any, str]) -> Tuple[List[Set[Any]], Set[Any], Dict[Any, int]]:
    dense_nodes = [n for n in graph.nodes() if builtup.get(n, 0.0) >= 0.52 and roles.get(n, "ordinary") != "bottleneck"]
    sub = graph.subgraph(dense_nodes).copy()
    comps = [set(comp) for comp in nx.connected_components(sub)] if sub.number_of_nodes() > 0 else []
    major = [comp for comp in comps if len(comp) >= 8]
    gateway_nodes: Set[Any] = set()
    cluster_id: Dict[Any, int] = {}
    for cid, comp in enumerate(major):
        for n in comp:
            cluster_id[n] = cid
        for n in comp:
            if any(nb not in comp for nb in graph.neighbors(n)):
                gateway_nodes.add(n)
    return major, gateway_nodes, cluster_id


def _select_depot(graph: nx.Graph, coords: Dict[Any, Tuple[float, float]], roles: Dict[Any, str], size_m: float) -> Tuple[Any, Dict[str, float]]:
    center_boxes = [0.40, 0.60]
    best_candidates: List[Any] = []
    selected_box = 0.60
    for frac in center_boxes:
        lo = 0.5 * (1.0 - frac) * float(size_m)
        hi = 0.5 * (1.0 + frac) * float(size_m)
        cand = []
        for n in graph.nodes():
            x, y = coords[n]
            deg = int(graph.degree(n))
            if deg < 2:
                continue
            if roles.get(n, "ordinary") in {"bottleneck"}:
                continue
            if not (lo <= x <= hi and lo <= y <= hi):
                continue
            cand.append(n)
        if cand:
            best_candidates = cand
            selected_box = frac
            break
    if not best_candidates:
        best_candidates = [n for n in graph.nodes() if int(graph.degree(n)) >= 2]
    best = None
    best_score = -1e18
    for n in best_candidates:
        x, y = coords[n]
        deg = int(graph.degree(n))
        role = roles.get(n, "ordinary")
        dirs = []
        for nb in graph.neighbors(n):
            nx_, ny_ = coords[nb]
            dirs.append(int(_orientation_bin(x, y, nx_, ny_)))
        dir_div = len(set(dirs))
        central_pen = abs(x - 0.5 * size_m) + abs(y - 0.5 * size_m)
        score = 0.0
        score += 1.6 * min(deg, 4)
        score += 1.2 * min(dir_div, 4)
        score += 2.0 if role == "arterial_junction" else 1.2 if role == "collector_junction" else 0.8 if role == "area_gateway" else 0.0
        score -= 0.00035 * central_pen
        if score > best_score:
            best_score = score
            best = n
    if best is None:
        best = next(iter(graph.nodes()))
    return best, {
        "depot_degree": float(graph.degree(best)),
        "depot_in_central_box": 1.0,
        "depot_backbone_access_score": float(best_score),
        "depot_box_fraction": float(selected_box),
    }


def _reindex_payload(
    graph: nx.Graph,
    coords: Dict[Any, Tuple[float, float]],
    node_data: Dict[Any, Dict[str, Any]],
    major_clusters: List[Set[Any]],
    gateway_nodes: Set[Any],
    cluster_id: Dict[Any, int],
    depot_node: Any,
) -> Dict[str, Any]:
    order = [depot_node] + [n for n in graph.nodes() if n != depot_node]
    remap = {old: idx for idx, old in enumerate(order)}
    node_xy = {int(remap[n]): (float(coords[n][0]), float(coords[n][1])) for n in order}
    adjacency: Dict[int, Set[int]] = {int(remap[n]): set() for n in order}
    edge_attrs: Dict[Tuple[int, int], Dict[str, Any]] = {}
    bridge_edges: Set[Tuple[int, int]] = set()
    for a, b, data in graph.edges(data=True):
        ia = int(remap[a])
        ib = int(remap[b])
        k = (min(ia, ib), max(ia, ib))
        adjacency[ia].add(ib)
        adjacency[ib].add(ia)
        edge_attrs[k] = dict(data)
        if bool(data.get("bridge_or_tunnel", False)):
            bridge_edges.add(k)
    nodes_raw: Dict[int, Dict[str, Any]] = {}
    for old in order:
        nid = int(remap[old])
        entry = dict(node_data[old])
        entry["node_id"] = nid
        nodes_raw[nid] = entry
    major_clusters_new = [[int(remap[n]) for n in sorted(comp, key=lambda x: remap[x])] for comp in major_clusters]
    gateway_new = sorted(int(remap[n]) for n in gateway_nodes if n in remap)
    cluster_id_new = {int(remap[n]): int(cid) for n, cid in cluster_id.items() if n in remap}
    return {
        "node_xy": node_xy,
        "adjacency": adjacency,
        "edge_attrs": edge_attrs,
        "nodes_raw": nodes_raw,
        "major_clusters": major_clusters_new,
        "gateway_nodes": gateway_new,
        "cluster_id_by_node": cluster_id_new,
        "bridge_edge_keys": sorted(bridge_edges),
    }


def _compute_summary(graph: nx.Graph, node_data: Dict[Any, Dict[str, Any]], edge_attrs: Dict[Tuple[int, int], Dict[str, Any]], depot_metrics: Dict[str, float], major_clusters: List[Set[Any]], gateway_nodes: Set[Any]) -> Dict[str, Any]:
    lengths = np.asarray([float(d.get("length_m", 0.0)) for d in edge_attrs.values()], dtype=np.float64)
    by_class = {"arterial": 0.0, "collector": 0.0, "local": 0.0}
    for d in edge_attrs.values():
        by_class[str(d.get("road_class", "collector"))] += float(d.get("length_m", 0.0))
    total_len = float(max(sum(by_class.values()), 1e-6))
    return {
        "node_count": int(graph.number_of_nodes()),
        "edge_count": int(graph.number_of_edges()),
        "road_class_share": {k: float(v / total_len) for k, v in by_class.items()},
        "bridge_edge_count": int(sum(1 for d in edge_attrs.values() if bool(d.get("bridge_or_tunnel", False)))),
        "depot_degree": float(depot_metrics.get("depot_degree", 0.0)),
        "depot_in_central_box": float(depot_metrics.get("depot_in_central_box", 0.0)),
        "depot_backbone_access_score": float(depot_metrics.get("depot_backbone_access_score", 0.0)),
        "major_cluster_count": int(len(major_clusters)),
        "gateway_count": int(len(gateway_nodes)),
        "arterial_mean_length_m": float(np.mean([float(d.get("length_m", 0.0)) for d in edge_attrs.values() if str(d.get("road_class", "collector")) == "arterial"]) if any(str(d.get("road_class", "collector")) == "arterial" for d in edge_attrs.values()) else 0.0),
        "collector_mean_length_m": float(np.mean([float(d.get("length_m", 0.0)) for d in edge_attrs.values() if str(d.get("road_class", "collector")) == "collector"]) if any(str(d.get("road_class", "collector")) == "collector" for d in edge_attrs.values()) else 0.0),
        "local_mean_length_m": float(np.mean([float(d.get("length_m", 0.0)) for d in edge_attrs.values() if str(d.get("road_class", "collector")) == "local"]) if any(str(d.get("road_class", "collector")) == "local" for d in edge_attrs.values()) else 0.0),
    }



def _load_prepared_clean_payload(cfg: EnvConfig) -> Optional[Dict[str, Any]]:
    graphml_path = Path(str(getattr(cfg, "real_case_prepared_graphml_path", "") or "").strip()).expanduser()
    if not bool(getattr(cfg, "real_case_use_prepared_clean_graph", False)):
        return None
    if not graphml_path.exists():
        return None

    g = nx.read_graphml(str(graphml_path))
    if g.number_of_nodes() == 0:
        return None
    if isinstance(g, (nx.MultiGraph, nx.MultiDiGraph)):
        g = nx.Graph(g)

    meta_json_path = Path(str(getattr(cfg, "real_case_poi_json_path", "") or "").strip()).expanduser()
    meta_blob: Dict[str, Any] = {}
    if meta_json_path.exists():
        try:
            meta_blob = json.loads(meta_json_path.read_text(encoding="utf-8"))
        except Exception:
            meta_blob = {}

    depot_old = int(meta_blob.get("depot_node_id", g.graph.get("depot_node_id", next(iter(g.nodes())))))
    order_old = [depot_old] + [int(n) for n in g.nodes() if int(n) != depot_old]
    remap = {int(old): idx for idx, old in enumerate(order_old)}

    nodes_raw: Dict[int, Dict[str, Any]] = {}
    node_xy: Dict[int, Tuple[float, float]] = {}
    adjacency: Dict[int, Set[int]] = {int(remap[int(n)]): set() for n in g.nodes()}
    for n, data in g.nodes(data=True):
        old = int(n)
        nid = int(remap[old])
        x = float(data.get("x", 0.0))
        y = float(data.get("y", 0.0))
        node_xy[nid] = (x, y)
        nodes_raw[nid] = {
            "x": x,
            "y": y,
            "elevation_m": float(data.get("elevation_m", 0.0)),
            "slope_norm": float(data.get("slope_norm", 0.0)),
            "node_role": str(data.get("node_role", "ordinary")),
            "builtup_intensity": float(data.get("builtup_intensity", 0.0)),
            "barrier_proximity": float(data.get("barrier_proximity", 0.0)),
            "poi_categories": str(data.get("poi_categories", "")),
            "quake_norm": float(data.get("quake_norm", 0.0)),
            "pga_g": float(data.get("pga_g", 0.0)),
            "pgv_cms": float(data.get("pgv_cms", 0.0)),
            "mmi": float(data.get("mmi", 0.0)),
        }

    edge_attrs: Dict[Tuple[int, int], Dict[str, Any]] = {}
    bridge_edge_keys: List[Tuple[int, int]] = []
    for u, v, data in g.edges(data=True):
        iu = int(remap[int(u)])
        iv = int(remap[int(v)])
        k = (min(iu, iv), max(iu, iv))
        adjacency[iu].add(iv)
        adjacency[iv].add(iu)
        edge_attrs[k] = {
            "road_class": str(data.get("road_class", "collector")),
            "length_m": float(data.get("length_m", _edge_length_from_nodes(*node_xy[iu], *node_xy[iv]))),
            "travel_speed_kph": float(data.get("travel_speed_kph", _default_speed_kph(str(data.get("road_class", "collector"))))),
            "lanes": int(float(data.get("lanes", 1))),
            "capacity_class": str(data.get("capacity_class", "medium")),
            "bridge_or_tunnel": bool(data.get("bridge_or_tunnel", False) in {True, 1, "1", "true", "True"}),
            "orientation_bin": int(float(data.get("orientation_bin", _orientation_bin(*node_xy[iu], *node_xy[iv])))),
            "builtup_exposure": float(data.get("builtup_exposure", 0.0)),
            "barrier_exposure": float(data.get("barrier_exposure", 0.0)),
            "roughness_norm": float(data.get("roughness_norm", 0.0)),
            "building_density_norm": float(data.get("building_density_norm", 0.0)),
            "infra_bottleneck_norm": float(data.get("infra_bottleneck_norm", 0.0)),
            "base_vulnerability": float(data.get("base_vulnerability", 0.0)),
        }
        if edge_attrs[k]["bridge_or_tunnel"]:
            bridge_edge_keys.append(k)

    # Prepared clean GraphML can preserve bridge tags while dropping the
    # derived risk attributes. Restore them deterministically so real-road
    # blockage keeps distinguishing bridges, gateways, and regular corridors.
    restored_edge_risk_attr_count = 0
    degree_map = {int(n): len(nbs) for n, nbs in adjacency.items()}
    for (a, b), data in edge_attrs.items():
        built_exp_calc = float(
            np.clip(
                0.5
                * (
                    float(nodes_raw.get(a, {}).get("builtup_intensity", 0.0))
                    + float(nodes_raw.get(b, {}).get("builtup_intensity", 0.0))
                ),
                0.0,
                1.0,
            )
        )
        barrier_exp_calc = float(
            np.clip(
                0.55
                * (
                    float(nodes_raw.get(a, {}).get("barrier_proximity", 0.0))
                    + float(nodes_raw.get(b, {}).get("barrier_proximity", 0.0))
                )
                * 0.5
                + (0.20 if bool(data.get("bridge_or_tunnel", False)) else 0.0),
                0.0,
                1.0,
            )
        )
        rc = str(data.get("road_class", "collector"))
        rough_calc = float(
            np.clip(
                0.18
                + 0.45
                * (1.0 if rc == "local" else 0.55 if rc == "collector" else 0.35)
                + 0.35 * barrier_exp_calc,
                0.0,
                1.0,
            )
        )
        infra_calc = float(
            np.clip(
                (0.85 if bool(data.get("bridge_or_tunnel", False)) else 0.25)
                + (
                    0.15
                    if degree_map.get(int(a), 0) <= 2 or degree_map.get(int(b), 0) <= 2
                    else 0.0
                ),
                0.0,
                1.0,
            )
        )
        v_base_calc = float(
            np.clip(
                (0.35 + 0.45 * barrier_exp_calc)
                * (1.0 + 0.5 * built_exp_calc)
                * (1.0 + infra_calc),
                0.0,
                1.0,
            )
        )
        zero_like = (
            float(data.get("builtup_exposure", 0.0)) <= 1e-9
            and float(data.get("barrier_exposure", 0.0)) <= 1e-9
            and float(data.get("roughness_norm", 0.0)) <= 1e-9
            and float(data.get("building_density_norm", 0.0)) <= 1e-9
            and float(data.get("infra_bottleneck_norm", 0.0)) <= 1e-9
            and float(data.get("base_vulnerability", 0.0)) <= 1e-9
        )
        if zero_like:
            restored_edge_risk_attr_count += 1
        if float(data.get("builtup_exposure", 0.0)) <= 1e-9:
            data["builtup_exposure"] = built_exp_calc
        if float(data.get("barrier_exposure", 0.0)) <= 1e-9:
            data["barrier_exposure"] = barrier_exp_calc
        if float(data.get("roughness_norm", 0.0)) <= 1e-9:
            data["roughness_norm"] = rough_calc
        if float(data.get("building_density_norm", 0.0)) <= 1e-9:
            data["building_density_norm"] = built_exp_calc
        if float(data.get("infra_bottleneck_norm", 0.0)) <= 1e-9:
            data["infra_bottleneck_norm"] = infra_calc
        if float(data.get("base_vulnerability", 0.0)) <= 1e-9:
            data["base_vulnerability"] = v_base_calc

    major_clusters_old = meta_blob.get("major_clusters", []) or []
    gateway_nodes_old = meta_blob.get("gateway_nodes", []) or []
    cluster_id_old = meta_blob.get("cluster_id_by_node", {}) or {}
    major_clusters = [[int(remap[int(n)]) for n in comp if int(n) in remap] for comp in major_clusters_old]
    gateway_nodes = [int(remap[int(n)]) for n in gateway_nodes_old if int(n) in remap]
    cluster_id_by_node = {int(remap[int(k)]): int(v) for k, v in cluster_id_old.items() if int(k) in remap}
    summary = dict(meta_blob.get("summary", {}))

    fixed_tasks: List[Dict[str, Any]] = []
    fixed_tasks_path = Path(
        str(getattr(cfg, "real_case_fixed_tasks_json_path", "") or "").strip()
    ).expanduser()
    if fixed_tasks_path.exists():
        fixed_blob = json.loads(fixed_tasks_path.read_text(encoding="utf-8"))
        fixed_items = fixed_blob.get("tasks", fixed_blob) if isinstance(fixed_blob, dict) else fixed_blob
        if not isinstance(fixed_items, list):
            raise ValueError(f"Fixed RB task manifest must contain a task list: {fixed_tasks_path}")
        for item in fixed_items:
            old_node = int(item.get("node_id", -1))
            if old_node not in remap:
                raise ValueError(f"Fixed RB task node {old_node} is absent from prepared graph")
            mapped = dict(item)
            mapped["node_id"] = int(remap[old_node])
            fixed_tasks.append(mapped)

    poi_items = []
    for item in meta_blob.get("poi_items", []):
        snapped_old = int(item.get("snapped_node", -1))
        if snapped_old not in remap:
            continue
        ni = dict(item)
        ni["snapped_node"] = int(remap[snapped_old])
        poi_items.append(ni)

    task_pools = {
        "bulk_builtup": [nid for nid, d in nodes_raw.items() if d["node_role"] in {"collector_junction", "ordinary"} and float(d["builtup_intensity"]) >= 0.45 and nid != 0],
        "bulk_gateway": [nid for nid in gateway_nodes if nid != 0],
        "bulk_peripheral": [nid for nid, d in nodes_raw.items() if 0.18 <= float(d["builtup_intensity"]) <= 0.60 and float(d["barrier_proximity"]) <= 0.72 and nid != 0],
        "timecritical_medical": sorted({int(item.get("snapped_node")) for item in poi_items if str(item.get("category", "")) in {"medical", "shelter"} and int(item.get("snapped_node", -1)) in nodes_raw and int(item.get("snapped_node")) != 0}),
        "timecritical_gateway": [nid for nid in gateway_nodes if nid != 0],
        "timecritical_hazard": [nid for nid, d in nodes_raw.items() if (float(d["barrier_proximity"]) >= 0.58 or d["node_role"] == "bottleneck") and nid != 0],
    }
    if not task_pools["timecritical_medical"]:
        task_pools["timecritical_medical"] = [nid for nid, d in nodes_raw.items() if d["node_role"] in {"collector_junction", "arterial_junction"} and float(d["builtup_intensity"]) >= 0.55 and nid != 0]

    real_case_meta = {
        "real_city_case": str(getattr(cfg, "real_city_case", "") or getattr(cfg, "real_case_name", "") or "dujiangyan"),
        "real_case_name": str(getattr(cfg, "real_case_name", "") or getattr(cfg, "real_city_case", "") or "dujiangyan"),
        "real_case_cleaning_profile": str(getattr(cfg, "real_case_cleaning_profile", "dujiangyan_v1_clean")),
        "real_case_task_sampling_profile": str(getattr(cfg, "real_case_task_sampling_profile", "dujiangyan_relief_v1")),
        "real_case_hazard_profile": str(getattr(cfg, "real_case_hazard_profile", "wenchuan_frontline_v1")),
        "real_case_bbox_mode": str(getattr(cfg, "real_case_bbox_mode", "center_size")),
        "real_case_size_m": float(getattr(cfg, "real_case_size_m", getattr(cfg, "map_size_m", 15000.0))),
        "depot_node_id": 0,
        "depot_degree": float(len(adjacency.get(0, set()))),
        "depot_in_central_box": float(summary.get("depot_in_central_box", 1.0)),
        "depot_backbone_access_score": float(summary.get("depot_backbone_access_score", len(adjacency.get(0, set())))),
        "major_clusters": major_clusters,
        "major_cluster_count": int(len(major_clusters)),
        "gateway_nodes": gateway_nodes,
        "cluster_id_by_node": cluster_id_by_node,
        "bridge_edge_keys": [tuple(map(int, e)) for e in bridge_edge_keys],
        "task_pools": task_pools,
        "poi_items": poi_items,
        "poi_counts": meta_blob.get("poi_counts", {}),
        "summary": summary,
        "fixed_tasks": fixed_tasks,
        "earthquake_field_mode": str(getattr(cfg, "earthquake_field_mode", "legacy_proxy")),
        "projected_crs": str(g.graph.get("crs", "")),
        "prepared_edge_risk_attrs_restored_count": int(restored_edge_risk_attr_count),
    }
    return {
        "node_xy": node_xy,
        "adjacency": adjacency,
        "edge_attrs": edge_attrs,
        "nodes_raw": nodes_raw,
        "major_clusters": major_clusters,
        "gateway_nodes": gateway_nodes,
        "cluster_id_by_node": cluster_id_by_node,
        "bridge_edge_keys": [tuple(map(int, e)) for e in bridge_edge_keys],
        "real_case_meta": real_case_meta,
    }

def build_real_city_case_payload(cfg: EnvConfig) -> Dict[str, Any]:
    prepared = _load_prepared_clean_payload(cfg)
    if prepared is not None:
        return prepared

    graphml_path = Path(str(cfg.osm_graphml_path)).expanduser()
    if not graphml_path.exists():
        raise FileNotFoundError(f"Real-city GraphML not found: {graphml_path}")

    g_raw = nx.read_graphml(str(graphml_path))
    if g_raw.number_of_nodes() == 0:
        raise ValueError(f"GraphML is empty: {graphml_path}")

    if isinstance(g_raw, (nx.MultiDiGraph, nx.MultiGraph)):
        edge_iter = g_raw.edges(keys=True, data=True)
    else:
        edge_iter = ((u, v, None, d) for u, v, d in g_raw.edges(data=True))

    node_coord_raw: Dict[Any, Tuple[float, float]] = {}
    for n, data in g_raw.nodes(data=True):
        x = _safe_float(data, ["x", "lon", "lng", "longitude", "X", "LONGITUDE"])
        y = _safe_float(data, ["y", "lat", "latitude", "Y", "LATITUDE"])
        if x is None or y is None:
            continue
        node_coord_raw[n] = (float(x), float(y))
    if not node_coord_raw:
        raise ValueError("GraphML does not contain usable x/y or lon/lat node coordinates")

    coord_vals = list(node_coord_raw.values())
    looks_geo = _looks_geographic(coord_vals)
    size_m = float(cfg.real_case_size_m if cfg.real_case_enabled else cfg.map_size_m)
    center_lon = float(cfg.real_case_center_lon)
    center_lat = float(cfg.real_case_center_lat)
    if looks_geo:
        node_xy = {n: _lonlat_to_local_xy(x, y, center_lon, center_lat, size_m) for n, (x, y) in node_coord_raw.items()}
    else:
        xs = np.asarray([xy[0] for xy in coord_vals], dtype=np.float64)
        ys = np.asarray([xy[1] for xy in coord_vals], dtype=np.float64)
        cx = float(np.median(xs))
        cy = float(np.median(ys))
        node_xy = {n: (float(x - cx + 0.5 * size_m), float(y - cy + 0.5 * size_m)) for n, (x, y) in node_coord_raw.items()}

    half = 0.5 * size_m
    keep_nodes = {
        n for n, (x, y) in node_xy.items()
        if (-0.05 * size_m) <= x <= (1.05 * size_m) and (-0.05 * size_m) <= y <= (1.05 * size_m)
    }
    if not keep_nodes:
        raise ValueError("No OSM nodes remain after real-case bbox crop")

    g = nx.Graph()
    for n in keep_nodes:
        g.add_node(n, **dict(g_raw.nodes[n]))

    edge_keep: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
    for u, v, _, data in edge_iter:
        if u not in keep_nodes or v not in keep_nodes or u == v:
            continue
        highway = _pick_highway(dict(data))
        if highway not in _ALLOWED_HIGHWAY:
            continue
        road_class = _road_class_from_highway(highway)
        x1, y1 = node_xy[u]
        x2, y2 = node_xy[v]
        length_m = float(_safe_float(dict(data), ["length", "length_m", "weight"]) or _edge_length_from_nodes(x1, y1, x2, y2))
        edge_data = {
            "road_class": road_class,
            "length_m": float(max(length_m, 1.0)),
            "travel_speed_kph": float(_parse_speed(dict(data).get("maxspeed"), road_class)),
            "lanes": int(_parse_lanes(dict(data).get("lanes"), road_class)),
            "capacity_class": _capacity_class(road_class, _parse_lanes(dict(data).get("lanes"), road_class)),
            "bridge_or_tunnel": bool(_is_bridge_or_tunnel(dict(data))),
            "orientation_bin": int(_orientation_bin(x1, y1, x2, y2)),
        }
        k = (u, v) if str(u) <= str(v) else (v, u)
        edge_keep[k] = _best_parallel_edge(edge_keep.get(k), edge_data)

    for (u, v), data in edge_keep.items():
        g.add_edge(u, v, **data)

    if g.number_of_nodes() == 0 or g.number_of_edges() == 0:
        raise ValueError("No usable real-city roads remain after cleaning filter")

    largest = max(nx.connected_components(g), key=len)
    g = g.subgraph(largest).copy()
    node_xy = {n: node_xy[n] for n in g.nodes()}

    g = _collapse_degree2_geometry(g, node_xy=node_xy, angle_thresh_deg=165.0)
    node_xy = {n: node_xy[n] for n in g.nodes()}

    dem = None
    dem_path = Path(str(cfg.dem_npy_path)).expanduser() if str(cfg.dem_npy_path).strip() else None
    if dem_path is not None and dem_path.exists() and dem_path.suffix.lower() == ".npy":
        try:
            arr = np.load(str(dem_path))
            if arr.ndim == 2:
                dem = arr.astype(np.float64)
        except Exception:
            dem = None

    slope_norm: Dict[Any, float] = {n: 0.0 for n in g.nodes()}
    elev_m: Dict[Any, float] = {n: 0.0 for n in g.nodes()}
    if dem is not None:
        gy, gx = np.gradient(dem)
        slope_field = np.hypot(gx, gy)
        ref = float(max(np.percentile(slope_field, 95), 1e-6))
        h, w = dem.shape
        for n, (x, y) in node_xy.items():
            u = float(np.clip(x / max(size_m, 1e-6), 0.0, 1.0))
            v = float(np.clip(y / max(size_m, 1e-6), 0.0, 1.0))
            j = int(round(u * (w - 1)))
            i = int(round(v * (h - 1)))
            elev_m[n] = float(dem[i, j])
            slope_norm[n] = float(np.clip(slope_field[i, j] / ref, 0.0, 1.0))

    builtup = _compute_pairwise_density(node_xy)
    barrier = _compute_barrier_proxy(node_xy, slope_norm, size_m)
    roles = _compute_node_roles(g, builtup, barrier)
    major_clusters, gateway_nodes, cluster_id = _detect_major_clusters(g, builtup, roles)
    depot_node, depot_metrics = _select_depot(g, node_xy, roles, size_m)
    roles[depot_node] = "depot_hub"

    node_data: Dict[Any, Dict[str, Any]] = {}
    for n in g.nodes():
        node_data[n] = {
            "x": float(node_xy[n][0]),
            "y": float(node_xy[n][1]),
            "elevation_m": float(elev_m.get(n, 0.0)),
            "slope_norm": float(slope_norm.get(n, 0.0)),
            "node_role": str(roles.get(n, "ordinary")),
            "builtup_intensity": float(builtup.get(n, 0.0)),
            "barrier_proximity": float(barrier.get(n, 0.0)),
        }

    for a, b, data in list(g.edges(data=True)):
        built_exp = float(np.clip(0.5 * (builtup.get(a, 0.0) + builtup.get(b, 0.0)), 0.0, 1.0))
        barrier_exp = float(np.clip(0.55 * (barrier.get(a, 0.0) + barrier.get(b, 0.0)) * 0.5 + (0.20 if bool(data.get("bridge_or_tunnel", False)) else 0.0), 0.0, 1.0))
        rc = str(data.get("road_class", "collector"))
        rough = float(np.clip(0.18 + 0.45 * (1.0 if rc == "local" else 0.55 if rc == "collector" else 0.35) + 0.35 * barrier_exp, 0.0, 1.0))
        infra = float(np.clip((0.85 if bool(data.get("bridge_or_tunnel", False)) else 0.25) + (0.15 if g.degree(a) <= 2 or g.degree(b) <= 2 else 0.0), 0.0, 1.0))
        v_base = float(np.clip((0.35 + 0.45 * barrier_exp) * (1.0 + 0.5 * built_exp) * (1.0 + infra), 0.0, 1.0))
        data["builtup_exposure"] = built_exp
        data["barrier_exposure"] = barrier_exp
        data["roughness_norm"] = rough
        data["building_density_norm"] = built_exp
        data["infra_bottleneck_norm"] = infra
        data["base_vulnerability"] = v_base

    payload = _reindex_payload(g, node_xy, node_data, major_clusters, gateway_nodes, cluster_id, depot_node)
    edge_attrs = payload["edge_attrs"]
    summary = _compute_summary(g, node_data, edge_attrs, depot_metrics, major_clusters, gateway_nodes)

    cluster_sizes = [len(comp) for comp in major_clusters]
    task_pools = {
        "bulk_builtup": [nid for nid, d in payload["nodes_raw"].items() if d["node_role"] in {"collector_junction", "ordinary"} and d["builtup_intensity"] >= 0.45 and nid != 0],
        "bulk_gateway": [nid for nid in payload["gateway_nodes"] if nid != 0],
        "bulk_peripheral": [nid for nid, d in payload["nodes_raw"].items() if 0.20 <= float(d["builtup_intensity"]) <= 0.60 and float(d["barrier_proximity"]) <= 0.65 and nid != 0],
        "timecritical_medical": [nid for nid, d in payload["nodes_raw"].items() if d["node_role"] in {"collector_junction", "arterial_junction"} and float(d["builtup_intensity"]) >= 0.55 and nid != 0],
        "timecritical_gateway": [nid for nid in payload["gateway_nodes"] if nid != 0],
        "timecritical_hazard": [nid for nid, d in payload["nodes_raw"].items() if (float(d["barrier_proximity"]) >= 0.58 or d["node_role"] == "bottleneck") and nid != 0],
    }

    payload["real_case_meta"] = {
        "real_city_case": str(cfg.real_city_case or cfg.real_case_name or "dujiangyan"),
        "real_case_name": str(cfg.real_case_name or cfg.real_city_case or "dujiangyan"),
        "real_case_cleaning_profile": str(cfg.real_case_cleaning_profile),
        "real_case_task_sampling_profile": str(cfg.real_case_task_sampling_profile),
        "real_case_hazard_profile": str(cfg.real_case_hazard_profile),
        "real_case_bbox_mode": str(cfg.real_case_bbox_mode),
        "real_case_size_m": float(size_m),
        "depot_node_id": 0,
        "depot_degree": float(summary["depot_degree"]),
        "depot_in_central_box": float(summary["depot_in_central_box"]),
        "depot_backbone_access_score": float(summary["depot_backbone_access_score"]),
        "major_clusters": payload["major_clusters"],
        "major_cluster_count": int(len(payload["major_clusters"])),
        "major_cluster_sizes": [int(x) for x in cluster_sizes],
        "gateway_nodes": payload["gateway_nodes"],
        "cluster_id_by_node": payload["cluster_id_by_node"],
        "bridge_edge_keys": payload["bridge_edge_keys"],
        "task_pools": task_pools,
        "summary": summary,
    }
    return payload
