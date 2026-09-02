from __future__ import annotations

import csv
import json
import math
import os
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import osmnx as ox
import yaml

from hetgat_hrl.core.mdp_spec import EnvConfig
from hetgat_hrl.envs.base_env import BaseHeteroDisasterEnv
from tools.prepare_dujiangyan_real_case import _build_clean_graph, _extract_pois, _plot, _save_graph


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_dujiangyan_quality_thresholds() -> Dict[str, Any]:
    return {
        "node_count_min": 120,
        "edge_count_min": 180,
        "largest_connected_component_ratio_min": 0.80,
        "depot_degree_min": 2,
        "depot_in_largest_component_required": True,
        "truck_reachable_routine_ratio_min": 0.80,
        "mean_task_to_nearest_road_m_max": 500.0,
        "node_count_max": 900,
        "edge_count_max": 1400,
        "require_disconnected_flag_false": True,
        "require_too_dense_flag_false": True,
    }


DUJIANGYAN_STANDARD_QUALITY_PROFILE = get_dujiangyan_quality_thresholds()


@dataclass
class CandidateWindow:
    candidate_id: str
    center_lat: float
    center_lon: float
    window_km: float
    center_label: str


@dataclass
class RealMapBuildSpec:
    scene_id: str
    place_label: str
    base_config_path: Path
    candidate_centers: List[Dict[str, Any]]
    candidate_window_km: List[float]
    output_config_path: Path
    output_cache_dir: Path
    cleaning_profile: str = "dujiangyan_standard"
    quality_profile: str = "dujiangyan_standard"
    task_profile: str = "urban_edge"
    seed_list: List[int] = None  # type: ignore[assignment]
    preferred_window_km: Optional[float] = None

    def __post_init__(self) -> None:
        if self.seed_list is None:
            self.seed_list = [0, 1, 2]
        if self.preferred_window_km is None:
            self.preferred_window_km = float(max(self.candidate_window_km) if self.candidate_window_km else 20.0)


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                cols.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _to_project_relative(path: Path) -> str:
    p = path.resolve()
    try:
        return str(p.relative_to(PROJECT_ROOT).as_posix())
    except Exception:
        return str(p.as_posix())


def _flatten_cfg(cfg_yaml: Dict[str, Any], seed: int) -> EnvConfig:
    merged: Dict[str, Any] = {}
    for section in ("env", "physics", "reward", "disturbance", "planner", "planner_toggles", "material_defaults"):
        part = cfg_yaml.get(section, {})
        if isinstance(part, dict):
            merged.update(part)
    merged["seed"] = int(seed)
    return EnvConfig(**merged)


def _km_offset(lat: float, lon: float, north_km: float, east_km: float) -> Tuple[float, float]:
    dlat = north_km / 111.32
    dlon = east_km / (111.32 * max(math.cos(math.radians(lat)), 1e-9))
    return float(lat + dlat), float(lon + dlon)


def _runtime_level(node_count: int, edge_count: int) -> str:
    if node_count >= 700 or edge_count >= 1100:
        return "high"
    if node_count >= 350 or edge_count >= 600:
        return "medium"
    return "low"


def _road_stats(g: nx.Graph, depot_node: Any) -> Dict[str, Any]:
    n = int(g.number_of_nodes())
    e = int(g.number_of_edges())
    deg = float(2.0 * e / max(n, 1))
    comps = list(nx.connected_components(g)) if n > 0 else []
    lcc_size = max((len(c) for c in comps), default=0)
    lcc_ratio = float(lcc_size / max(n, 1))
    dead_ends = int(sum(1 for x in g.nodes() if int(g.degree(x)) <= 1))
    dead_end_ratio = float(dead_ends / max(n, 1))
    def _resolve_node_key(node_like: Any) -> Any:
        # GraphML ids may load as str while metadata often stores int.
        if node_like in g:
            return node_like
        s = str(node_like)
        if s in g:
            return s
        try:
            i = int(node_like)
            if i in g:
                return i
            si = str(i)
            if si in g:
                return si
        except Exception:
            pass
        return None

    dnode = _resolve_node_key(depot_node)
    depot_degree = int(g.degree(dnode)) if (dnode is not None and dnode in g) else 0
    depot_in_lcc = bool(any(dnode in c for c in comps if len(c) == lcc_size)) if dnode is not None else False
    lengths = [float(d.get("length_m", 0.0)) for _, _, d in g.edges(data=True)]
    road_total_km = float(sum(lengths) / 1000.0)
    bridge_tunnel = int(sum(1 for _, _, d in g.edges(data=True) if bool(d.get("bridge_or_tunnel", False))))
    return {
        "cleaned_node_count": n,
        "cleaned_edge_count": e,
        "road_length_total_km": road_total_km,
        "largest_connected_component_ratio": lcc_ratio,
        "average_degree": deg,
        "dead_end_node_count": dead_ends,
        "dead_end_ratio": dead_end_ratio,
        "depot_degree": depot_degree,
        "depot_in_largest_component": depot_in_lcc,
        "bridge_or_tunnel_count": bridge_tunnel,
        "median_edge_length_m": float(np.median(np.asarray(lengths, dtype=np.float64)) if lengths else 0.0),
        "p90_edge_length_m": float(np.percentile(np.asarray(lengths, dtype=np.float64), 90.0) if lengths else 0.0),
    }


def _evaluate_env_metrics(cfg_yaml: Dict[str, Any], seed: int = 0) -> Dict[str, Any]:
    env = BaseHeteroDisasterEnv(_flatten_cfg(cfg_yaml, seed=seed))
    topo = env.topology
    tasks = list(env.state.tasks.values())
    routine = [t for t in tasks if str(getattr(t, "task_class", "")) == "routine_bulk"]
    emer = [t for t in tasks if str(getattr(t, "task_class", "")) == "time_critical_lightweight"]
    trucks = [aid for aid, s in env.state.agents.items() if str(getattr(s, "kind", "")).endswith("TRUCK")]
    uavs = [aid for aid, s in env.state.agents.items() if str(getattr(s, "kind", "")).endswith("UAV")]

    def truck_reach(task: Any) -> bool:
        for tid in trucks:
            st = env.state.agents.get(str(tid), None)
            if st is None or st.node is None:
                continue
            if topo.path_exists(int(st.node), int(task.demand_node), ignore_blocked=True):
                return True
        return False

    r_reach = int(sum(1 for t in routine if truck_reach(t)))
    e_reach = int(sum(1 for t in emer if truck_reach(t)))
    immediate = 0
    eventual = 0
    forced_ids = set(getattr(env, "_forced_island_task_ids", set()) or set())
    max_sortie = float(max(getattr(env.cfg, "uav_max_sortie_m", 2400.0), 1.0))
    for t in emer:
        ok_now = False
        for uid in uavs:
            fn = getattr(env, "_uav_docked_task_actionable_now", None)
            if callable(fn) and bool(fn(str(uid), t)):
                ok_now = True
                break
        if ok_now:
            immediate += 1
            eventual += 1
            continue
        txy = topo.nodes[int(t.demand_node)]
        maybe = False
        for tid in trucks:
            st = env.state.agents.get(str(tid), None)
            if st is None or st.node is None:
                continue
            sxy = topo.nodes[int(st.node)]
            d = float(np.hypot(float(txy.x) - float(sxy.x), float(txy.y) - float(sxy.y)))
            if d <= float(max_sortie * 1.2):
                maybe = True
                break
        if maybe:
            eventual += 1

    total_e = max(len(emer), 1)
    uav_only = int(sum(1 for t in emer if not truck_reach(t)))
    return {
        "routine_task_count": int(len(routine)),
        "emergency_task_count": int(len(emer)),
        "routine_truck_reachable_ratio": float(r_reach / max(len(routine), 1)),
        "emergency_truck_reachable_ratio": float(e_reach / total_e),
        "truck_reachable_routine_count": int(r_reach),
        "truck_reachable_emergency_count": int(e_reach),
        "uav_only_emergency_candidate_ratio": float(uav_only / total_e),
        "forced_island_emergency_count": int(sum(1 for t in emer if str(t.task_id) in forced_ids)),
        "uav_eventual_serviceable_emergency_ratio": float(eventual / total_e),
        "uav_immediate_launchable_emergency_ratio": float(immediate / total_e),
        "uav_eventual_serviceable_emergency_count": int(eventual),
        "uav_immediate_launchable_emergency_count": int(immediate),
    }


def _acceptance(candidate_row: Dict[str, Any], profile: Dict[str, Any]) -> Tuple[bool, str]:
    fail: List[str] = []
    if int(candidate_row.get("cleaned_node_count", 0)) < int(profile["node_count_min"]):
        fail.append("node_count<min")
    if int(candidate_row.get("cleaned_edge_count", 0)) < int(profile["edge_count_min"]):
        fail.append("edge_count<min")
    if float(candidate_row.get("largest_connected_component_ratio", 0.0)) < float(profile["largest_connected_component_ratio_min"]):
        fail.append("lcc_ratio<min")
    if int(candidate_row.get("depot_degree", 0)) < int(profile["depot_degree_min"]):
        fail.append("depot_degree<min")
    if bool(profile["depot_in_largest_component_required"]) and not bool(candidate_row.get("depot_in_largest_component", False)):
        fail.append("depot_not_in_lcc")
    if float(candidate_row.get("routine_truck_reachable_ratio", 0.0)) < float(profile["truck_reachable_routine_ratio_min"]):
        fail.append("routine_truck_reachable<min")
    if float(candidate_row.get("mean_task_to_nearest_road_m", 0.0)) > float(profile["mean_task_to_nearest_road_m_max"]):
        fail.append("mean_task_to_road>max")
    if bool(profile["require_too_dense_flag_false"]) and bool(candidate_row.get("too_dense_flag", False)):
        fail.append("too_dense")
    if bool(profile["require_disconnected_flag_false"]) and bool(candidate_row.get("disconnected_flag", False)):
        fail.append("disconnected")
    return (len(fail) == 0, "|".join(fail))


def _score_candidate(row: Dict[str, Any], preferred_window_km: float) -> float:
    s = 1000.0 if bool(row.get("accepted_flag", False)) else 0.0
    s += 150.0 * float(row.get("routine_truck_reachable_ratio", 0.0))
    s += 90.0 * float(row.get("emergency_truck_reachable_ratio", 0.0))
    s += 60.0 * float(row.get("uav_eventual_serviceable_emergency_ratio", 0.0))
    s += 10.0 * float(row.get("depot_degree", 0.0))
    s -= 10.0 * abs(float(row.get("window_km", preferred_window_km)) - float(preferred_window_km))
    if str(row.get("estimated_runtime_level", "")) == "high":
        s -= 120.0
    if bool(row.get("too_sparse_flag", False)):
        s -= 200.0
    if bool(row.get("too_dense_flag", False)):
        s -= 180.0
    if bool(row.get("disconnected_flag", False)):
        s -= 260.0
    return float(s)


def _resolve_center(center_spec: Dict[str, Any]) -> Tuple[str, float, float]:
    label = str(center_spec.get("label", "")).strip()
    if "lat" in center_spec and "lon" in center_spec:
        return label, float(center_spec["lat"]), float(center_spec["lon"])
    if not label:
        raise ValueError("candidate center needs label or lat/lon")
    # Geocode fallback (WGS84 via OSM/Nominatim through osmnx)
    lat, lon = ox.geocode(label)
    return label, float(lat), float(lon)


def _enumerate_candidate_windows(spec: RealMapBuildSpec) -> List[CandidateWindow]:
    out: List[CandidateWindow] = []
    seen = set()
    for center in spec.candidate_centers:
        c_label, c_lat, c_lon = _resolve_center(center)
        for wk in spec.candidate_window_km:
            cid = f"{c_label.replace(' ', '_').replace('/', '_').replace(',', '')}_w{int(round(float(wk)))}"
            k = (round(c_lat, 6), round(c_lon, 6), round(float(wk), 3))
            if k in seen:
                continue
            seen.add(k)
            out.append(CandidateWindow(candidate_id=cid, center_lat=float(c_lat), center_lon=float(c_lon), window_km=float(wk), center_label=c_label))
    return out


def _download_drive_graph(center_lat: float, center_lon: float, size_m: float, raw_graphml: Path) -> Tuple[int, int]:
    raw_graphml.parent.mkdir(parents=True, exist_ok=True)
    if raw_graphml.exists():
        g = nx.read_graphml(str(raw_graphml))
        return int(g.number_of_nodes()), int(g.number_of_edges())
    dist = int(max(1000.0, float(size_m) * 0.82))
    ox.settings.use_cache = True
    ox.settings.log_console = False
    overpass_url = str(os.environ.get("OSMNX_OVERPASS_URL", "")).strip()
    if overpass_url:
        ox.settings.overpass_url = overpass_url
    g = ox.graph_from_point((float(center_lat), float(center_lon)), dist=dist, network_type="drive", simplify=True, retain_all=True)
    ox.save_graphml(g, filepath=str(raw_graphml))
    return int(len(g.nodes())), int(len(g.edges()))


def evaluate_candidate_window(
    candidate: CandidateWindow,
    spec: RealMapBuildSpec,
    base_cfg_yaml: Dict[str, Any],
    run_dir: Path,
) -> Dict[str, Any]:
    env_base = base_cfg_yaml.get("env", {}) if isinstance(base_cfg_yaml.get("env", {}), dict) else {}
    scene_dir = PROJECT_ROOT / spec.output_cache_dir
    cdir = scene_dir / "candidates" / candidate.candidate_id
    raw_graphml = cdir / "raw.graphml"
    clean_graphml = cdir / "cleaned.graphml"
    poi_json = cdir / "poi_candidates.json"
    preview_png = cdir / "preview.png"
    summary_json = cdir / "candidate_summary.json"

    row: Dict[str, Any] = {
        "scene_id": spec.scene_id,
        "candidate_id": candidate.candidate_id,
        "center_label": candidate.center_label,
        "center_lat": float(candidate.center_lat),
        "center_lon": float(candidate.center_lon),
        "window_km": float(candidate.window_km),
        "raw_graphml_path": _to_project_relative(raw_graphml),
        "cleaned_graphml_path": _to_project_relative(clean_graphml),
        "poi_json_path": _to_project_relative(poi_json),
        "cleaning_profile": spec.cleaning_profile,
        "quality_profile": spec.quality_profile,
        "task_profile": spec.task_profile,
        "error": "",
    }
    try:
        raw_n, raw_e = _download_drive_graph(candidate.center_lat, candidate.center_lon, candidate.window_km * 1000.0, raw_graphml)
        row["raw_node_count"] = int(raw_n)
        row["raw_edge_count"] = int(raw_e)

        g_clean, node_xy, meta = _build_clean_graph(
            raw_graphml=raw_graphml,
            center_lon=float(candidate.center_lon),
            center_lat=float(candidate.center_lat),
            size_m=float(candidate.window_km) * 1000.0,
            min_leaf_edge_m=float(env_base.get("real_case_min_leaf_edge_m", 320.0)),
            postmerge_leaf_edge_m=float(env_base.get("real_case_postmerge_leaf_edge_m", 420.0)),
            merge_cell_m=float(env_base.get("real_case_junction_merge_cell_m", 280.0)),
            chain_collapse_angle_deg=float(env_base.get("real_case_chain_collapse_angle_deg", 145.0)),
        )
        poi = _extract_pois(
            center_lat=float(candidate.center_lat),
            center_lon=float(candidate.center_lon),
            size_m=float(candidate.window_km) * 1000.0,
            node_xy=node_xy,
            graph=g_clean,
        )
        cdir.mkdir(parents=True, exist_ok=True)
        _save_graph(g_clean, node_xy, meta, poi, clean_graphml)
        _plot(g_clean, node_xy, meta, poi, float(candidate.window_km) * 1000.0, preview_png)

        poi_blob = {
            "depot_node_id": int(meta.get("depot_node", 0)),
            "gateway_nodes": [int(x) for x in meta.get("gateway_nodes", [])],
            "major_clusters": [[int(n) for n in comp] for comp in meta.get("clusters", [])],
            "cluster_id_by_node": {str(k): int(v) for k, v in (meta.get("cluster_id_by_node", {}) or {}).items()},
            "poi_counts": dict(poi.get("counts", {})),
            "poi_items": poi.get("items", []),
            "summary": dict(meta.get("summary", {})),
        }
        poi_json.write_text(json.dumps(poi_blob, ensure_ascii=False, indent=2), encoding="utf-8")

        road = _road_stats(g_clean, meta.get("depot_node", 0))
        row.update(road)
        row["candidate_depot_node"] = int(meta.get("depot_node", 0))
        snap_d = [float(x.get("snap_distance_m", 0.0)) for x in poi.get("items", [])]
        row["mean_task_to_nearest_road_m"] = float(np.mean(np.asarray(snap_d, dtype=np.float64)) if snap_d else 0.0)
        row["max_task_to_nearest_road_m"] = float(np.max(np.asarray(snap_d, dtype=np.float64)) if snap_d else 0.0)

        cfg_eval = json.loads(json.dumps(base_cfg_yaml))
        env_cfg = cfg_eval.setdefault("env", {})
        env_cfg["real_case_enabled"] = True
        env_cfg["map_source"] = "osm_dem"
        env_cfg["real_case_center_lat"] = float(candidate.center_lat)
        env_cfg["real_case_center_lon"] = float(candidate.center_lon)
        env_cfg["real_case_size_m"] = float(candidate.window_km) * 1000.0
        env_cfg["real_case_use_prepared_clean_graph"] = True
        env_cfg["real_case_prepared_graphml_path"] = _to_project_relative(clean_graphml)
        env_cfg["real_case_poi_json_path"] = _to_project_relative(poi_json)
        env_cfg["osm_graphml_path"] = _to_project_relative(raw_graphml)
        row.update(_evaluate_env_metrics(cfg_eval, seed=int(spec.seed_list[0])))

        n = int(row.get("cleaned_node_count", 0))
        e = int(row.get("cleaned_edge_count", 0))
        row["estimated_runtime_level"] = _runtime_level(n, e)
        q = DUJIANGYAN_STANDARD_QUALITY_PROFILE
        row["too_sparse_flag"] = bool(n < int(q["node_count_min"]) or e < int(q["edge_count_min"]))
        row["too_dense_flag"] = bool(n > int(q["node_count_max"]) or e > int(q["edge_count_max"]))
        row["disconnected_flag"] = bool(
            float(row.get("largest_connected_component_ratio", 0.0)) < float(q["largest_connected_component_ratio_min"])
        )
        row["primary_secondary_road_ratio"] = float(
            (sum(1 for _, _, d in g_clean.edges(data=True) if str(d.get("road_class", "")) in {"arterial", "collector"}))
            / max(sum(1 for _ in g_clean.edges()), 1)
        )
        ok, reason = _acceptance(row, q)
        row["accepted_flag"] = bool(ok)
        row["rejection_reason"] = reason
        row["score"] = _score_candidate(row, preferred_window_km=float(spec.preferred_window_km or 20.0))
    except Exception as exc:
        row["accepted_flag"] = False
        row["rejection_reason"] = f"build_error:{type(exc).__name__}"
        row["score"] = -1e9
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc(limit=3)

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def select_best_candidate(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not candidates:
        return {}
    accepted = [r for r in candidates if bool(r.get("accepted_flag", False))]
    pool = accepted if accepted else candidates
    pool = sorted(
        pool,
        key=lambda r: (
            float(r.get("score", -1e9)),
            -abs(float(r.get("cleaned_node_count", 0)) - 320.0),
            -abs(float(r.get("cleaned_edge_count", 0)) - 460.0),
        ),
        reverse=True,
    )
    return pool[0]


def write_real_map_config(selected_candidate: Dict[str, Any], spec: RealMapBuildSpec, base_cfg_yaml: Dict[str, Any]) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg_yaml))
    env = cfg.setdefault("env", {})
    rb = cfg.setdefault("real_map_builder", {})
    m = cfg.setdefault("metadata", {})

    final_dir = PROJECT_ROOT / spec.output_cache_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    city_stub = spec.scene_id.lower().replace("-", "_")
    final_graph = final_dir / f"{city_stub}_drive_cleaned.graphml"
    final_poi = final_dir / f"{city_stub}_poi_candidates.json"

    src_graph = PROJECT_ROOT / str(selected_candidate.get("cleaned_graphml_path", ""))
    src_poi = PROJECT_ROOT / str(selected_candidate.get("poi_json_path", ""))
    if src_graph.exists():
        shutil.copyfile(str(src_graph), str(final_graph))
    if src_poi.exists():
        shutil.copyfile(str(src_poi), str(final_poi))

    cfg["source"] = spec.output_config_path.stem
    env["real_case_enabled"] = True
    env["map_source"] = "osm_dem"
    env["real_city_case"] = spec.scene_id.lower().replace("-", "_")
    env["real_case_name"] = f"{env['real_city_case']}_real_RC_capacity_matched"
    env["real_case_center_lat"] = float(selected_candidate.get("center_lat", env.get("real_case_center_lat", 30.0)))
    env["real_case_center_lon"] = float(selected_candidate.get("center_lon", env.get("real_case_center_lon", 103.0)))
    env["real_case_size_m"] = float(selected_candidate.get("window_km", 20.0)) * 1000.0
    env["real_case_use_prepared_clean_graph"] = True
    env["real_case_prepared_graphml_path"] = _to_project_relative(final_graph)
    env["real_case_poi_json_path"] = _to_project_relative(final_poi)
    if str(selected_candidate.get("raw_graphml_path", "")).strip():
        env["osm_graphml_path"] = str(selected_candidate.get("raw_graphml_path", ""))

    rb["scene_id"] = spec.scene_id
    rb["place_label"] = spec.place_label
    rb["center_lat"] = float(selected_candidate.get("center_lat", rb.get("center_lat", 30.0)))
    rb["center_lon"] = float(selected_candidate.get("center_lon", rb.get("center_lon", 103.0)))
    rb["preferred_window_km"] = float(selected_candidate.get("window_km", rb.get("preferred_window_km", 20.0)))
    rb["candidate_window_km"] = [float(x) for x in spec.candidate_window_km]
    rb["candidate_centers"] = [dict(x) for x in spec.candidate_centers]
    m["scene_id"] = spec.scene_id
    m["place_label"] = spec.place_label
    return cfg


def export_real_map_preview(selected_candidate: Dict[str, Any], output_png: Path) -> None:
    src_preview = (PROJECT_ROOT / str(selected_candidate.get("cleaned_graphml_path", ""))).parent / "preview.png"
    output_png.parent.mkdir(parents=True, exist_ok=True)
    if src_preview.exists():
        shutil.copyfile(str(src_preview), str(output_png))


def build_real_map_from_spec(spec: RealMapBuildSpec, results_dir: Path) -> Dict[str, Any]:
    base_cfg = _read_yaml(spec.base_config_path)
    candidates = _enumerate_candidate_windows(spec)
    rows: List[Dict[str, Any]] = []
    for c in candidates:
        rows.append(evaluate_candidate_window(candidate=c, spec=spec, base_cfg_yaml=base_cfg, run_dir=results_dir))

    selected = select_best_candidate(rows)
    selected_cfg = write_real_map_config(selected, spec=spec, base_cfg_yaml=base_cfg)
    _write_yaml(spec.output_config_path, selected_cfg)

    preview_path = results_dir / "previews" / f"{spec.scene_id}_road_network.png"
    export_real_map_preview(selected, preview_path)

    quality_row = {
        "scene_id": spec.scene_id,
        "place_label": spec.place_label,
        "selected_candidate_id": str(selected.get("candidate_id", "")),
        "selected_center_lat": float(selected.get("center_lat", 0.0)),
        "selected_center_lon": float(selected.get("center_lon", 0.0)),
        "selected_window_km": float(selected.get("window_km", 0.0)),
        "node_count": int(selected.get("cleaned_node_count", 0) or 0),
        "edge_count": int(selected.get("cleaned_edge_count", 0) or 0),
        "largest_connected_component_ratio": float(selected.get("largest_connected_component_ratio", 0.0)),
        "depot_degree": int(selected.get("depot_degree", 0) or 0),
        "depot_in_largest_component": bool(selected.get("depot_in_largest_component", False)),
        "too_sparse_flag": bool(selected.get("too_sparse_flag", False)),
        "too_dense_flag": bool(selected.get("too_dense_flag", False)),
        "disconnected_flag": bool(selected.get("disconnected_flag", False)),
        "accepted_flag": bool(selected.get("accepted_flag", False)),
        "rejection_reason": str(selected.get("rejection_reason", "")),
        "config_path": _to_project_relative(spec.output_config_path),
        "preview_png": _to_project_relative(preview_path),
        "cleaning_profile": spec.cleaning_profile,
        "quality_profile": spec.quality_profile,
    }
    return {"spec": spec, "candidate_rows": rows, "selected_row": selected, "quality_row": quality_row}


def build_report(path: Path, run_name: str, results: Sequence[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("# REAL RC MAPS SAME AS DJY BUILD REPORT")
    lines.append("")
    lines.append(f"- run_name: `{run_name}`")
    lines.append("- R-DJ-C already existed; this run adds R-YA-C, R-CQ-C, R-KM-C, R-HZ-C.")
    lines.append("- all new cities reuse one single pipeline: dujiangyan_standard cleaning + quality + task profile.")
    lines.append("- no ERC/rolling algorithm change; no formal algorithm comparison was run.")
    lines.append("")
    for r in results:
        q = r["quality_row"]
        lines.append(f"## {q['scene_id']} ({q['place_label']})")
        lines.append(f"- selected center: ({q['selected_center_lat']:.6f}, {q['selected_center_lon']:.6f})")
        lines.append(f"- selected window_km: {q['selected_window_km']:.1f}")
        lines.append(f"- node_count/edge_count: {q['node_count']}/{q['edge_count']}")
        lines.append(f"- largest_component_ratio: {q['largest_connected_component_ratio']:.3f}")
        lines.append(f"- depot_degree: {q['depot_degree']}")
        lines.append(f"- accepted_flag: {q['accepted_flag']}")
        if str(q.get("rejection_reason", "")).strip():
            lines.append(f"- rejection_reason: `{q['rejection_reason']}`")
        lines.append("")
    lines.append("## Method")
    lines.append("- unified profile: cleaning_profile=dujiangyan_standard, quality_profile=dujiangyan_standard, task_profile=urban_edge.")
    lines.append("- if a city fails, standards are not relaxed; alternate windows/cities should be used.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tables(
    run_dir: Path,
    results: Sequence[Dict[str, Any]],
    candidate_table_name: str,
    selected_table_name: str,
    quality_table_name: str,
) -> Dict[str, str]:
    tables = run_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    candidate_rows = [x for r in results for x in r.get("candidate_rows", [])]
    selected_rows = [
        {"scene_id": r["quality_row"]["scene_id"], "place_label": r["quality_row"]["place_label"], **dict(r.get("selected_row", {}))}
        for r in results
    ]
    quality_rows = [dict(r["quality_row"]) for r in results]
    candidate_path = tables / candidate_table_name
    selected_path = tables / selected_table_name
    quality_path = tables / quality_table_name
    _write_csv(candidate_path, candidate_rows)
    _write_csv(selected_path, selected_rows)
    _write_csv(quality_path, quality_rows)
    return {
        "candidate_table": _to_project_relative(candidate_path),
        "selected_table": _to_project_relative(selected_path),
        "quality_table": _to_project_relative(quality_path),
    }
