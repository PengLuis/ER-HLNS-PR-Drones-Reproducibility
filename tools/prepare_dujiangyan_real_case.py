from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

KEEP_HIGHWAYS = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
}

POI_TAGS = {
    "amenity": [
        "hospital",
        "clinic",
        "doctors",
        "school",
        "college",
        "university",
        "kindergarten",
        "parking",
        "shelter",
        "community_centre",
    ],
    "leisure": ["park", "stadium", "sports_centre", "pitch"],
    "emergency": ["assembly_point"],
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
            import ast
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
        "motorway_link": 1,
        "trunk": 2,
        "trunk_link": 3,
        "primary": 4,
        "primary_link": 5,
        "secondary": 6,
        "secondary_link": 7,
        "tertiary": 8,
        "tertiary_link": 9,
        "unclassified": 10,
        "residential": 11,
        "service": 12,
    }
    vals.sort(key=lambda x: priority.get(x, 99))
    return vals[0]


def _road_class_from_highway(highway: str) -> str:
    hw = str(highway).strip().lower()
    if hw in {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link"}:
        return "arterial"
    if hw in {"secondary", "secondary_link", "tertiary", "tertiary_link"}:
        return "collector"
    return "local"


def _parse_lanes(value: Any, road_class: str) -> int:
    vals = _as_listish(value)
    for v in vals:
        try:
            return max(1, int(round(float(v))))
        except Exception:
            continue
    return 2 if road_class != "local" else 1


def _parse_speed(value: Any, road_class: str) -> float:
    defaults = {"arterial": 55.0, "collector": 38.0, "local": 25.0}
    vals = _as_listish(value)
    for v in vals:
        raw = str(v).lower().replace("km/h", "").replace("kph", "").replace("mph", "")
        try:
            out = float(raw)
            if "mph" in str(v).lower():
                out *= 1.60934
            return max(out, 5.0)
        except Exception:
            continue
    return defaults.get(road_class, 35.0)


def _is_bridge_or_tunnel(data: Dict[str, Any]) -> bool:
    bridge = str(data.get("bridge", "")).strip().lower()
    tunnel = str(data.get("tunnel", "")).strip().lower()
    return bridge not in {"", "no", "false", "0", "none"} or tunnel not in {"", "no", "false", "0", "none"}


def _lonlat_to_local_xy(lon: float, lat: float, center_lon: float, center_lat: float, size_m: float) -> Tuple[float, float]:
    lat_scale = 111320.0
    lon_scale = 111320.0 * math.cos(math.radians(center_lat))
    x = (float(lon) - float(center_lon)) * lon_scale + 0.5 * float(size_m)
    y = (float(lat) - float(center_lat)) * lat_scale + 0.5 * float(size_m)
    return float(x), float(y)


def _orientation_bin(ax: float, ay: float, bx: float, by: float) -> int:
    ang = float((math.degrees(math.atan2(by - ay, bx - ax)) + 360.0) % 180.0)
    return int(round(ang / 22.5)) % 8


def _best_edge(old: Optional[Dict[str, Any]], new: Dict[str, Any]) -> Dict[str, Any]:
    if old is None:
        return new
    rank = {"arterial": 0, "collector": 1, "local": 2}
    ro = rank.get(str(old.get("road_class", "collector")), 9)
    rn = rank.get(str(new.get("road_class", "collector")), 9)
    if rn < ro:
        return new
    if rn > ro:
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


def _edge_length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.hypot(b[0] - a[0], b[1] - a[1]))


def _read_cfg(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _flatten_cfg(data: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for section in ("env", "physics", "reward", "disturbance"):
        part = data.get(section, {})
        if isinstance(part, dict):
            merged.update(part)
    return merged


def _build_clean_graph(
    raw_graphml: Path,
    center_lon: float,
    center_lat: float,
    size_m: float,
    min_leaf_edge_m: float,
    postmerge_leaf_edge_m: float,
    merge_cell_m: float,
    chain_collapse_angle_deg: float,
) -> Tuple[nx.Graph, Dict[Any, Tuple[float, float]], Dict[str, Any]]:
    g_raw = nx.read_graphml(str(raw_graphml))
    node_xy: Dict[Any, Tuple[float, float]] = {}
    for n, data in g_raw.nodes(data=True):
        lon = _safe_float(data, ["x", "lon", "lng", "longitude", "X"])
        lat = _safe_float(data, ["y", "lat", "latitude", "Y"])
        if lon is None or lat is None:
            continue
        node_xy[n] = _lonlat_to_local_xy(lon, lat, center_lon, center_lat, size_m)
    bbox_nodes = {n for n, (x, y) in node_xy.items() if 0.0 <= x <= size_m and 0.0 <= y <= size_m}

    g = nx.Graph()
    for n in bbox_nodes:
        lon = _safe_float(g_raw.nodes[n], ["x", "lon", "lng", "longitude", "X"])
        lat = _safe_float(g_raw.nodes[n], ["y", "lat", "latitude", "Y"])
        g.add_node(n, lon=float(lon), lat=float(lat))

    if isinstance(g_raw, (nx.MultiGraph, nx.MultiDiGraph)):
        edge_iter = g_raw.edges(keys=True, data=True)
    else:
        edge_iter = ((u, v, None, d) for u, v, d in g_raw.edges(data=True))
    edge_keep: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
    filtered_counts: Dict[str, int] = {}
    for u, v, _, data in edge_iter:
        if u not in bbox_nodes or v not in bbox_nodes or u == v:
            continue
        hw = _pick_highway(dict(data))
        filtered_counts[hw] = filtered_counts.get(hw, 0) + 1
        if hw not in KEEP_HIGHWAYS:
            continue
        rc = _road_class_from_highway(hw)
        a = node_xy[u]
        b = node_xy[v]
        lanes = _parse_lanes(data.get("lanes"), rc)
        length_m = float(_safe_float(dict(data), ["length", "length_m", "weight"]) or _edge_length(a, b))
        ed = {
            "highway": hw,
            "road_class": rc,
            "length_m": max(length_m, 1.0),
            "travel_speed_kph": _parse_speed(data.get("maxspeed"), rc),
            "lanes": lanes,
            "capacity_class": "high" if rc == "arterial" else "medium" if rc == "collector" else "low",
            "bridge_or_tunnel": _is_bridge_or_tunnel(dict(data)),
            "orientation_bin": _orientation_bin(*a, *b),
        }
        key = (u, v) if str(u) <= str(v) else (v, u)
        edge_keep[key] = _best_edge(edge_keep.get(key), ed)
    for (u, v), data in edge_keep.items():
        g.add_edge(u, v, **data)
    largest = max(nx.connected_components(g), key=len)
    g = g.subgraph(largest).copy()
    node_xy = {n: node_xy[n] for n in g.nodes()}

    def prune_leaf_stubs(graph: nx.Graph, threshold_m: float) -> nx.Graph:
        changed = True
        while changed:
            changed = False
            for n in list(graph.nodes()):
                if graph.degree(n) != 1:
                    continue
                nb = next(iter(graph.neighbors(n)))
                if bool(graph.edges[n, nb].get("bridge_or_tunnel", False)):
                    continue
                if float(graph.edges[n, nb].get("length_m", 0.0)) < threshold_m:
                    graph.remove_node(n)
                    changed = True
                    break
        return graph

    def collapse_degree2(graph: nx.Graph, angle_thresh: float) -> nx.Graph:
        changed = True
        while changed:
            changed = False
            for n in list(graph.nodes()):
                if n not in graph or graph.degree(n) != 2:
                    continue
                a, b = list(graph.neighbors(n))
                if a == b:
                    continue
                e1 = dict(graph.edges[n, a])
                e2 = dict(graph.edges[n, b])
                if bool(e1.get("bridge_or_tunnel", False) or e2.get("bridge_or_tunnel", False)):
                    continue
                ang = _line_angle_deg(node_xy[a], node_xy[n], node_xy[b])
                total_len = float(e1.get("length_m", 0.0)) + float(e2.get("length_m", 0.0))
                if ang < angle_thresh and total_len > 300.0:
                    continue
                rc = e1.get("road_class") if e1.get("road_class") == e2.get("road_class") else ("arterial" if "arterial" in {e1.get("road_class"), e2.get("road_class")} else "collector")
                merged = {
                    "highway": e1.get("highway", e2.get("highway", "tertiary")),
                    "road_class": rc,
                    "length_m": total_len,
                    "travel_speed_kph": min(float(e1.get("travel_speed_kph", 35.0)), float(e2.get("travel_speed_kph", 35.0))),
                    "lanes": max(int(e1.get("lanes", 1)), int(e2.get("lanes", 1))),
                    "capacity_class": e1.get("capacity_class", e2.get("capacity_class", "medium")),
                    "bridge_or_tunnel": False,
                    "orientation_bin": _orientation_bin(*node_xy[a], *node_xy[b]),
                }
                if graph.has_edge(a, b):
                    merged = _best_edge(dict(graph.edges[a, b]), merged)
                graph.add_edge(a, b, **merged)
                graph.remove_node(n)
                changed = True
                break
        return graph

    def merge_near_junctions(graph: nx.Graph, cell_m: float) -> Tuple[nx.Graph, Dict[Any, Tuple[float, float]]]:
        buckets: Dict[Tuple[int, int], List[Any]] = {}
        for n, (x, y) in node_xy.items():
            buckets.setdefault((int(x // cell_m), int(y // cell_m)), []).append(n)
        rep_map: Dict[Any, Any] = {}
        new_xy: Dict[Any, Tuple[float, float]] = {}
        new_graph = nx.Graph()
        for bucket_nodes in buckets.values():
            if len(bucket_nodes) == 1:
                rep = bucket_nodes[0]
                rep_map[rep] = rep
                new_xy[rep] = node_xy[rep]
                new_graph.add_node(rep, **dict(graph.nodes[rep]))
                continue
            rep = max(bucket_nodes, key=lambda n: (graph.degree(n), sum(float(graph.edges[n, nb].get("length_m", 0.0)) for nb in graph.neighbors(n))))
            arr = np.asarray([node_xy[n] for n in bucket_nodes], dtype=np.float64)
            rep_map.update({n: rep for n in bucket_nodes})
            new_xy[rep] = (float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1])))
            node_attr = dict(graph.nodes[rep])
            node_attr["merge_count"] = int(len(bucket_nodes))
            new_graph.add_node(rep, **node_attr)
        for u, v, data in graph.edges(data=True):
            ru = rep_map[u]
            rv = rep_map[v]
            if ru == rv:
                continue
            a = new_xy[ru]
            b = new_xy[rv]
            nd = dict(data)
            nd["length_m"] = _edge_length(a, b)
            nd["orientation_bin"] = _orientation_bin(*a, *b)
            if new_graph.has_edge(ru, rv):
                nd = _best_edge(dict(new_graph.edges[ru, rv]), nd)
            new_graph.add_edge(ru, rv, **nd)
        return new_graph, new_xy

    def prune_collinear(graph: nx.Graph, angle_deg: float = 165.0) -> nx.Graph:
        removed = True
        while removed:
            removed = False
            for b in list(graph.nodes()):
                nbs = list(graph.neighbors(b))
                if len(nbs) < 2:
                    continue
                for i in range(len(nbs)):
                    for j in range(i + 1, len(nbs)):
                        a, c = nbs[i], nbs[j]
                        if not graph.has_edge(a, c):
                            continue
                        ang = _line_angle_deg(node_xy[a], node_xy[b], node_xy[c])
                        if ang < angle_deg:
                            continue
                        eab = graph.edges[a, b]
                        ebc = graph.edges[b, c]
                        eac = graph.edges[a, c]
                        if bool(eac.get("bridge_or_tunnel", False)):
                            continue
                        if str(eac.get("road_class", "collector")) not in {str(eab.get("road_class", "collector")), str(ebc.get("road_class", "collector"))}:
                            continue
                        if float(eac.get("length_m", 0.0)) >= 0.92 * (float(eab.get("length_m", 0.0)) + float(ebc.get("length_m", 0.0))):
                            graph.remove_edge(a, c)
                            removed = True
                            break
                    if removed:
                        break
                if removed:
                    break
        return graph

    g = prune_leaf_stubs(g, min_leaf_edge_m)
    node_xy = {n: node_xy[n] for n in g.nodes()}
    g = collapse_degree2(g, float(chain_collapse_angle_deg))
    node_xy = {n: node_xy[n] for n in g.nodes()}
    g, node_xy = merge_near_junctions(g, merge_cell_m)
    g = prune_leaf_stubs(g, postmerge_leaf_edge_m)
    node_xy = {n: node_xy[n] for n in g.nodes()}
    g = collapse_degree2(g, float(chain_collapse_angle_deg))
    node_xy = {n: node_xy[n] for n in g.nodes()}
    g = prune_collinear(g)
    if not nx.is_connected(g):
        g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
        node_xy = {n: node_xy[n] for n in g.nodes()}

    density: Dict[Any, float] = {}
    pts = np.asarray([node_xy[n] for n in g.nodes()], dtype=np.float64)
    keys = list(g.nodes())
    if len(keys) > 0:
        for i, n in enumerate(keys):
            d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
            density[n] = float((np.sum(d <= 500.0) - 1.0) / max(1.0, np.percentile(np.sum(np.hypot(pts[:, 0][:, None] - pts[:, 0][None, :], pts[:, 1][:, None] - pts[:, 1][None, :]) <= 500.0, axis=1) - 1.0, 95)))
    builtup = {n: float(np.clip(density.get(n, 0.0), 0.0, 1.0)) for n in g.nodes()}
    barrier = {}
    for n, (x, y) in node_xy.items():
        xn = x / size_m
        yn = y / size_m
        barrier[n] = float(np.clip(0.65 * (1.0 - xn) + 0.35 * yn, 0.0, 1.0))

    roles: Dict[Any, str] = {}
    for n in g.nodes():
        deg = int(g.degree(n))
        cls = [str(g.edges[n, nb].get("road_class", "collector")) for nb in g.neighbors(n)]
        arterial = sum(1 for c in cls if c == "arterial")
        collector = sum(1 for c in cls if c == "collector")
        bridge = any(bool(g.edges[n, nb].get("bridge_or_tunnel", False)) for nb in g.neighbors(n))
        if bridge:
            roles[n] = "bottleneck"
        elif deg >= 3 and arterial >= 2:
            roles[n] = "arterial_junction"
        elif deg >= 3 and arterial + collector >= 2:
            roles[n] = "collector_junction"
        elif builtup[n] >= 0.45 and arterial + collector >= 1:
            roles[n] = "area_gateway"
        else:
            roles[n] = "ordinary"

    dense_nodes = [n for n in g.nodes() if builtup[n] >= 0.45 and roles[n] != "bottleneck"]
    sub = g.subgraph(dense_nodes).copy()
    clusters = [sorted(list(comp)) for comp in nx.connected_components(sub) if len(comp) >= 6] if sub.number_of_nodes() else []
    gateway_nodes: Set[Any] = set()
    cluster_id_by_node: Dict[int, int] = {}
    for cid, comp in enumerate(clusters):
        for n in comp:
            cluster_id_by_node[int(n)] = int(cid)
        comp_set = set(comp)
        external = [n for n in comp if any(nb not in comp_set for nb in g.neighbors(n))]
        gateway_nodes.update(external)
    central_lo = size_m * 0.30
    central_hi = size_m * 0.70
    candidates = [n for n, (x, y) in node_xy.items() if central_lo <= x <= central_hi and central_lo <= y <= central_hi and g.degree(n) >= 2 and roles[n] != "bottleneck"]
    if not candidates:
        central_lo = size_m * 0.20
        central_hi = size_m * 0.80
        candidates = [n for n, (x, y) in node_xy.items() if central_lo <= x <= central_hi and central_lo <= y <= central_hi and g.degree(n) >= 2 and roles[n] != "bottleneck"]
    depot_node = max(candidates or list(g.nodes()), key=lambda n: (g.degree(n), sum(1 for nb in g.neighbors(n) if str(g.edges[n, nb].get("road_class", "collector")) in {"arterial", "collector"}), -abs(node_xy[n][0] - 0.5 * size_m) - abs(node_xy[n][1] - 0.5 * size_m)))
    roles[depot_node] = "depot_hub"

    summary = {
        "raw_graph_nodes": int(g_raw.number_of_nodes()),
        "raw_graph_edges": int(g_raw.number_of_edges()),
        "cleaned_node_count": int(g.number_of_nodes()),
        "cleaned_edge_count": int(g.number_of_edges()),
        "major_cluster_count": int(len(clusters)),
        "gateway_count": int(len(gateway_nodes)),
        "depot_degree": int(g.degree(depot_node)),
        "depot_node_id": str(depot_node),
        "kept_highway_classes": sorted(KEEP_HIGHWAYS),
        "filtered_highway_counts_raw": filtered_counts,
    }
    return g, node_xy, {
        "builtup": builtup,
        "barrier": barrier,
        "roles": roles,
        "clusters": clusters,
        "gateway_nodes": sorted(gateway_nodes),
        "cluster_id_by_node": cluster_id_by_node,
        "depot_node": depot_node,
        "summary": summary,
    }


def _extract_pois(center_lat: float, center_lon: float, size_m: float, node_xy: Dict[Any, Tuple[float, float]], graph: nx.Graph) -> Dict[str, Any]:
    overpass_url = str(os.environ.get("OSMNX_OVERPASS_URL", "")).strip()
    if overpass_url:
        ox.settings.overpass_url = overpass_url
    try:
        gdf = ox.features_from_point((center_lat, center_lon), tags=POI_TAGS, dist=int(size_m * 0.60))
    except Exception as exc:
        return {"error": str(exc), "items": [], "counts": {}}
    coords = np.asarray([node_xy[n] for n in graph.nodes()], dtype=np.float64)
    node_order = list(graph.nodes())
    items: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for _, row in gdf.iterrows():
        geom = getattr(row, "geometry", None)
        if geom is None:
            continue
        try:
            rp = geom.representative_point()
            lon = float(rp.x)
            lat = float(rp.y)
        except Exception:
            continue
        x, y = _lonlat_to_local_xy(lon, lat, center_lon, center_lat, size_m)
        if not (0.0 <= x <= size_m and 0.0 <= y <= size_m):
            continue
        category = None
        amenity = str(row.get("amenity", "") or "").strip().lower()
        leisure = str(row.get("leisure", "") or "").strip().lower()
        emergency = str(row.get("emergency", "") or "").strip().lower()
        if amenity in {"hospital", "clinic", "doctors"}:
            category = "medical"
        elif amenity in {"school", "college", "university", "kindergarten"}:
            category = "school"
        elif amenity in {"parking"}:
            category = "parking"
        elif amenity in {"shelter", "community_centre"} or emergency in {"assembly_point"}:
            category = "shelter"
        elif leisure in {"park", "stadium", "sports_centre", "pitch"}:
            category = "park"
        if category is None:
            continue
        if len(node_order) == 0:
            continue
        d = np.hypot(coords[:, 0] - x, coords[:, 1] - y)
        idx = int(np.argmin(d))
        snap_node = int(node_order[idx])
        item = {
            "name": str(row.get("name", "") or f"{category}_{len(items)}"),
            "category": category,
            "lon": lon,
            "lat": lat,
            "x": float(x),
            "y": float(y),
            "snapped_node": snap_node,
            "snap_distance_m": float(d[idx]),
        }
        items.append(item)
        counts[category] = counts.get(category, 0) + 1
    return {"items": items, "counts": counts}


def _save_graph(graph: nx.Graph, node_xy: Dict[Any, Tuple[float, float]], meta: Dict[str, Any], poi: Dict[str, Any], out_graphml: Path) -> None:
    out = nx.Graph()
    poi_by_node: Dict[int, List[str]] = {}
    for item in poi.get("items", []):
        poi_by_node.setdefault(int(item["snapped_node"]), []).append(str(item["category"]))
    for n in graph.nodes():
        out.add_node(
            int(n),
            x=float(node_xy[n][0]),
            y=float(node_xy[n][1]),
            lon=float(graph.nodes[n].get("lon", 0.0)),
            lat=float(graph.nodes[n].get("lat", 0.0)),
            node_role=str(meta["roles"].get(n, "ordinary")),
            builtup_intensity=float(meta["builtup"].get(n, 0.0)),
            barrier_proximity=float(meta["barrier"].get(n, 0.0)),
            is_gateway=int(n in set(meta["gateway_nodes"])),
            cluster_id=int(meta["cluster_id_by_node"].get(int(n), -1)),
            poi_categories="|".join(sorted(set(poi_by_node.get(int(n), [])))),
        )
    for u, v, data in graph.edges(data=True):
        out.add_edge(int(u), int(v), **data)
    out.graph["depot_node_id"] = int(meta["depot_node"])
    out.graph["major_cluster_count"] = int(len(meta["clusters"]))
    nx.write_graphml(out, out_graphml)


def _plot(graph: nx.Graph, node_xy: Dict[Any, Tuple[float, float]], meta: Dict[str, Any], poi: Dict[str, Any], size_m: float, out_png: Path) -> None:
    fig = plt.figure(figsize=(11, 11), dpi=160)
    ax = fig.add_subplot(111)
    colors = {"arterial": "#1d3557", "collector": "#457b9d", "local": "#a8dadc"}
    widths = {"arterial": 1.6, "collector": 1.0, "local": 0.6}
    for u, v, data in graph.edges(data=True):
        a = node_xy[u]
        b = node_xy[v]
        rc = str(data.get("road_class", "collector"))
        ax.plot([a[0], b[0]], [a[1], b[1]], color=colors.get(rc, "#999999"), linewidth=widths.get(rc, 0.8), alpha=0.9)
        if bool(data.get("bridge_or_tunnel", False)):
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#f4a261", linewidth=widths.get(rc, 0.8) + 0.8, alpha=0.9)
    for comp in meta["clusters"]:
        pts = [node_xy[n] for n in comp]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=15, c="#90be6d", alpha=0.18)
    gtw = [node_xy[n] for n in meta["gateway_nodes"] if n in node_xy]
    if gtw:
        ax.scatter([p[0] for p in gtw], [p[1] for p in gtw], s=24, c="#577590", marker="s", alpha=0.9, label="gateway")
    dep = node_xy[meta["depot_node"]]
    ax.scatter([dep[0]], [dep[1]], s=180, c="#111111", marker="*", edgecolors="white", linewidths=0.8, label="depot")
    cat_colors = {"medical": "#e63946", "school": "#ffb703", "park": "#2a9d8f", "parking": "#6d597a", "shelter": "#f77f00"}
    grouped: Dict[str, List[Tuple[float, float]]] = {}
    for item in poi.get("items", []):
        grouped.setdefault(str(item["category"]), []).append((float(item["x"]), float(item["y"])))
    for cat, pts in grouped.items():
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=34, c=cat_colors.get(cat, "#333333"), alpha=0.85, label=cat)
    ax.set_xlim(0.0, size_m)
    ax.set_ylim(0.0, size_m)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title("Dujiangyan cleaned real-road execution graph")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare cleaned Dujiangyan real-road benchmark graph and POI cache.")
    ap.add_argument("--config", type=str, default="configs/real_dujiangyan_LB.yaml")
    ap.add_argument("--raw-graphml", type=str, default="")
    ap.add_argument("--out-graphml", type=str, default="data/real_city/dujiangyan/dujiangyan_drive_cleaned.graphml")
    ap.add_argument("--out-poi-json", type=str, default="data/real_city/dujiangyan/dujiangyan_poi_candidates.json")
    ap.add_argument("--out-summary-json", type=str, default="data/real_city/dujiangyan/dujiangyan_cleaning_summary.json")
    ap.add_argument("--out-summary-md", type=str, default="data/real_city/dujiangyan/dujiangyan_cleaning_summary.md")
    ap.add_argument("--out-png", type=str, default="data/real_city/dujiangyan/dujiangyan_cleaning_overview.png")
    args = ap.parse_args()

    cfg = _flatten_cfg(_read_cfg(PROJECT_ROOT / args.config))
    raw_graphml = PROJECT_ROOT / (args.raw_graphml or cfg.get("osm_graphml_path", "data/real_city/dujiangyan/dujiangyan_drive.graphml"))
    center_lon = float(cfg.get("real_case_center_lon", 103.61941))
    center_lat = float(cfg.get("real_case_center_lat", 30.99825))
    size_m = float(cfg.get("real_case_size_m", 15000.0))
    min_leaf = float(cfg.get("real_case_min_leaf_edge_m", 260.0))
    post_leaf = float(cfg.get("real_case_postmerge_leaf_edge_m", 360.0))
    merge_cell = float(cfg.get("real_case_junction_merge_cell_m", 260.0))
    collapse_angle = float(cfg.get("real_case_chain_collapse_angle_deg", 145.0))

    graph, node_xy, meta = _build_clean_graph(
        raw_graphml,
        center_lon,
        center_lat,
        size_m,
        min_leaf,
        post_leaf,
        merge_cell,
        collapse_angle,
    )
    poi = _extract_pois(center_lat, center_lon, size_m, node_xy, graph)

    out_graphml = PROJECT_ROOT / args.out_graphml
    out_poi_json = PROJECT_ROOT / args.out_poi_json
    out_summary_json = PROJECT_ROOT / args.out_summary_json
    out_summary_md = PROJECT_ROOT / args.out_summary_md
    out_png = PROJECT_ROOT / args.out_png
    for p in [out_graphml, out_poi_json, out_summary_json, out_summary_md, out_png]:
        p.parent.mkdir(parents=True, exist_ok=True)

    summary = dict(meta["summary"])
    summary.update({
        "cleaning_profile": str(cfg.get("real_case_cleaning_profile", "dujiangyan_v1")).strip(),
        "target_node_lt_500": bool(graph.number_of_nodes() < 500),
        "target_edge_lt_1000": bool(graph.number_of_edges() < 1000),
        "poi_counts": dict(poi.get("counts", {})),
    })

    _save_graph(graph, node_xy, meta, poi, out_graphml)
    out_poi_json.write_text(json.dumps({
        "depot_node_id": int(meta["depot_node"]),
        "gateway_nodes": [int(x) for x in meta["gateway_nodes"]],
        "major_clusters": [[int(n) for n in comp] for comp in meta["clusters"]],
        "cluster_id_by_node": {str(k): int(v) for k, v in meta["cluster_id_by_node"].items()},
        "poi_counts": dict(poi.get("counts", {})),
        "poi_items": poi.get("items", []),
        "summary": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    out_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md_lines = ["# Dujiangyan real-road cleaning summary", ""]
    for k, v in summary.items():
        md_lines.append(f"- `{k}`: `{v}`")
    out_summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    _plot(graph, node_xy, meta, poi, size_m, out_png)
    print(json.dumps({
        "graphml": str(out_graphml),
        "poi_json": str(out_poi_json),
        "summary_json": str(out_summary_json),
        "png": str(out_png),
        "cleaned_nodes": int(graph.number_of_nodes()),
        "cleaned_edges": int(graph.number_of_edges()),
        "poi_counts": dict(poi.get("counts", {})),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
