from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from hetgat_hrl.core.mdp_spec import EnvConfig
from hetgat_hrl.core.topology import GraphTopology


@dataclass
class NodeHazard:
    rain: float
    wind: float
    quake: float


class GlobalHazardField:
    """
    Continuous geophysical field over 2D space.
    - Quake: single-source hypocentral attenuation (static residual field).
    - Wind: parametric vortex storm (moving center, Rankine-like profile).
    - Rain: eyewall Gaussian rainband coupled to storm center.
    """

    def __init__(
        self,
        map_bounds: Tuple[float, float, float, float],
        dt_seconds: float,
        seed: int,
        base_rainfall_mmh: float,
        base_wind_mps: float,
        enable_weather: bool = True,
    ):
        self.rng = np.random.default_rng(seed)
        self.dt = float(max(dt_seconds, 1e-6))
        self.x_min, self.x_max, self.y_min, self.y_max = map_bounds
        self.enable_weather = bool(enable_weather)

        sx = float(max(self.x_max - self.x_min, 1.0))
        sy = float(max(self.y_max - self.y_min, 1.0))
        diag = float(np.hypot(sx, sy))
        margin = float(0.20 * diag)

        # 1) Quake: single epicenter + hypocentral attenuation.
        self.quake_epicenter = (
            float(self.rng.uniform(self.x_min - margin, self.x_max + margin)),
            float(self.rng.uniform(self.y_min - margin, self.y_max + margin)),
        )
        self.quake_depth = 300.0
        self.quake_max_intensity = 1.0 if self.enable_weather else 0.0
        self.quake_gamma = 1.5

        # 2) Storm: moving vortex center.
        cx_lo, cx_hi = self.x_min + 0.15 * sx, self.x_max - 0.15 * sx
        cy_lo, cy_hi = self.y_min + 0.15 * sy, self.y_max - 0.15 * sy
        if cx_lo >= cx_hi:
            cx_lo, cx_hi = self.x_min, self.x_max
        if cy_lo >= cy_hi:
            cy_lo, cy_hi = self.y_min, self.y_max
        self.storm_center = np.array(
            [
                float(self.rng.uniform(cx_lo, cx_hi)),
                float(self.rng.uniform(cy_lo, cy_hi)),
            ],
            dtype=np.float64,
        )

        drift_speed = float(
            5.0 if base_wind_mps <= 0.0 else np.clip(0.6 * base_wind_mps, 2.0, 8.0)
        )
        drift_theta = float(self.rng.uniform(0.0, 2.0 * np.pi))
        self.storm_velocity = np.array(
            [drift_speed * np.cos(drift_theta), drift_speed * np.sin(drift_theta)],
            dtype=np.float64,
        )
        self.boundary_pad = float(0.25 * diag)

        self.rmw = float(np.clip(0.08 * diag, 250.0, 900.0))
        self.v_max = float(np.clip(max(base_wind_mps, 0.0) * 3.8, 0.0, 30.0))
        self.wind_decay_alpha = 0.55
        self.inflow_angle = float(np.deg2rad(20.0))

        # 3) Rainband coupled to storm.
        self.rain_max = float(np.clip(max(base_rainfall_mmh, 0.0) * 2.5, 0.0, 35.0))
        self.rain_sigma = float(np.clip(0.06 * diag, 180.0, 420.0))

    def step(self) -> None:
        if not self.enable_weather:
            return
        self.storm_center = self.storm_center + self.storm_velocity * self.dt
        # Smooth reflection at expanded boundary so storm keeps traversing map.
        low = np.array(
            [self.x_min - self.boundary_pad, self.y_min - self.boundary_pad],
            dtype=np.float64,
        )
        high = np.array(
            [self.x_max + self.boundary_pad, self.y_max + self.boundary_pad],
            dtype=np.float64,
        )
        for i in range(2):
            if self.storm_center[i] < low[i]:
                self.storm_center[i] = low[i] + (low[i] - self.storm_center[i])
                self.storm_velocity[i] *= -1.0
            elif self.storm_center[i] > high[i]:
                self.storm_center[i] = high[i] - (self.storm_center[i] - high[i])
                self.storm_velocity[i] *= -1.0

    def get_hazard_at(self, x: float, y: float) -> Tuple[float, float, float, float, float]:
        # 1) Quake attenuation from single hypocenter.
        dx_q = float(x - self.quake_epicenter[0])
        dy_q = float(y - self.quake_epicenter[1])
        r_epi = float(np.hypot(dx_q, dy_q))
        r_hypo = float(np.hypot(r_epi, self.quake_depth))
        if r_hypo <= 1e-9:
            quake = float(self.quake_max_intensity)
        else:
            quake = float(
                self.quake_max_intensity
                * (self.quake_depth / r_hypo) ** self.quake_gamma
            )
        quake = float(np.clip(quake, 0.0, 1.0))

        # 2) Parametric vortex wind.
        dx_s = float(x - self.storm_center[0])
        dy_s = float(y - self.storm_center[1])
        r = float(np.hypot(dx_s, dy_s))
        if (not self.enable_weather) or self.v_max <= 1e-9 or r <= 1e-6:
            vx, vy, v_mag = 0.0, 0.0, 0.0
        else:
            if r <= self.rmw:
                v_prof = float(self.v_max * (r / max(self.rmw, 1e-6)))
            else:
                v_prof = float(self.v_max * (self.rmw / max(r, 1e-6)) ** self.wind_decay_alpha)
            theta = float(np.arctan2(dy_s, dx_s))
            # CCW tangential + inward inflow angle.
            wind_dir = theta + 0.5 * np.pi + self.inflow_angle
            vx = float(v_prof * np.cos(wind_dir) + 0.5 * self.storm_velocity[0])
            vy = float(v_prof * np.sin(wind_dir) + 0.5 * self.storm_velocity[1])
            v_mag = float(np.hypot(vx, vy))

        # 3) Eyewall Gaussian rainband around RMW.
        if (not self.enable_weather) or self.rain_max <= 1e-9:
            rain = 0.0
        else:
            rain = float(
                self.rain_max
                * np.exp(-((r - self.rmw) ** 2) / (2.0 * max(self.rain_sigma**2, 1e-6)))
            )
        rain = float(max(rain, 0.0))
        return quake, v_mag, vx, vy, rain


class DynamicHazardField:
    """
    Wrapper that keeps existing env API stable while switching internals to
    a continuous global physical hazard field.
    """

    def __init__(
        self,
        topo: GraphTopology,
        seed: int,
        stochastic_weather: bool = True,
        cfg: Optional[EnvConfig] = None,
    ):
        self.topo = topo
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.stochastic_weather = bool(stochastic_weather)
        self.step_index = 0
        self._weather_cache_step: int = -1
        self._weather_cache: Dict[Tuple[int, int], NodeHazard] = {}
        self._field_hazard_cache_step: int = -1
        self._field_hazard_cache: Dict[Tuple[int, int], Tuple[float, float, float, float, float]] = {}
        self._nearest_node_xy_cache: Dict[Tuple[int, int], int] = {}

        self.base_rainfall_mmh = float(cfg.base_rainfall_mmh if cfg is not None else 12.0)
        self.base_wind_mps = float(cfg.base_wind_mps if cfg is not None else 6.0)

        xs = np.array([n.x for n in self.topo.nodes.values()], dtype=np.float64)
        ys = np.array([n.y for n in self.topo.nodes.values()], dtype=np.float64)
        x_min = float(xs.min()) if xs.size else 0.0
        x_max = float(xs.max()) if xs.size else 3000.0
        y_min = float(ys.min()) if ys.size else 0.0
        y_max = float(ys.max()) if ys.size else 3000.0

        dt_s = float(cfg.dt_seconds if cfg is not None else 20.0)
        enable_weather = bool(self.stochastic_weather and (self.base_wind_mps > 0.0 or self.base_rainfall_mmh > 0.0))
        self.field = GlobalHazardField(
            map_bounds=(x_min, x_max, y_min, y_max),
            dt_seconds=dt_s,
            seed=seed + 1000,
            base_rainfall_mmh=self.base_rainfall_mmh,
            base_wind_mps=self.base_wind_mps,
            enable_weather=enable_weather,
        )
        self.real_case_enabled = bool(getattr(cfg, "real_case_enabled", False)) and str(getattr(cfg, "map_source", "")).strip().lower() == "osm_dem"
        self.real_case_hazard_profile = str(getattr(cfg, "real_case_hazard_profile", "")).strip().lower()
        self.earthquake_field_mode = str(getattr(cfg, "earthquake_field_mode", "legacy_proxy")).strip().lower()
        self.rb_road_damage_mode = str(getattr(cfg, "rb_road_damage_mode", "legacy_mixed")).strip().lower()
        self._river_corridor_x = float(0.5 * (x_min + x_max))
        bridge_x = []
        for src, nbs in self.topo.adjacency.items():
            for dst in nbs:
                if src >= dst:
                    continue
                eattr = self.topo.edge_attr(int(src), int(dst))
                if bool(getattr(eattr, "bridge_or_tunnel", False)):
                    a = self.topo.nodes[int(src)]
                    b = self.topo.nodes[int(dst)]
                    bridge_x.append(0.5 * (float(a.x) + float(b.x)))
        if bridge_x:
            self._river_corridor_x = float(np.median(np.asarray(bridge_x, dtype=np.float64)))
        if self.real_case_enabled and self.real_case_hazard_profile == "wenchuan_frontline_v1":
            sx = float(max(x_max - x_min, 1.0))
            sy = float(max(y_max - y_min, 1.0))
            self.field.quake_epicenter = (float(x_min + 0.18 * sx), float(y_min + 0.78 * sy))
            self.field.quake_gamma = 1.35
            self.field.storm_center = np.array([float(self._river_corridor_x), float(y_min + 0.68 * sy)], dtype=np.float64)
            self.field.storm_velocity = np.array([1.2, -2.4], dtype=np.float64)
            self.field.rmw = float(np.clip(0.10 * np.hypot(sx, sy), 450.0, 1500.0))
            self.field.rain_sigma = float(np.clip(0.10 * np.hypot(sx, sy), 380.0, 1500.0))

        self.node_hazard: Dict[int, NodeHazard] = {}
        self.epicenter_node = 0
        self.last_percolation_phase: str = "aggressive"
        self.last_lambda: float = float(cfg.lambda_aggressive) if cfg is not None else 0.012
        self.last_pmacro_mean: float = 0.0
        self.last_pstep_mean: float = 0.0
        self.last_edge_pstep: Dict[Tuple[int, int], float] = {}
        self.last_bldg_mean: float = 0.0
        self.last_infra_mean: float = 0.0
        self.last_length_norm_mean: float = 0.0
        self.blocked_edge_age_steps: Dict[Tuple[int, int], int] = {}
        self.newly_blocked_edge_count_step: int = 0
        self.newly_blocked_edge_count_total: int = 0
        self.blocked_edge_count: int = 0
        self.blocked_edge_count_stochastic: int = 0
        self.blocked_edge_count_forced_island: int = 0
        self.blocked_edge_count_total: int = 0
        self.blocked_edge_count_stochastic_step: int = 0
        self.blocked_edge_count_forced_island_step: int = 0
        self.blocked_edge_count_total_step: int = 0
        self.blocked_ratio_stochastic: float = 0.0
        self.blocked_ratio_forced_island: float = 0.0
        self.blocked_ratio_total: float = 0.0
        self.blockage_curve_B_inf: float = 0.0
        self.blockage_curve_tau_steps: float = 1.0
        self.blockage_target_ratio_step: float = 0.0
        self.blockage_target_ratio_stochastic_step: float = 0.0
        self.blockage_gap_step: float = 0.0
        self.blockage_global_gate_step: float = 0.0
        self.blockage_current_ratio_stochastic_step: float = 0.0
        self.ever_blocked_edge_keys: set[Tuple[int, int]] = set()
        self.forced_island_edge_keys: set[Tuple[int, int]] = set()
        self.nonreopen_edge_keys: set[Tuple[int, int]] = set()
        self.stochastic_blocked_edge_keys: set[Tuple[int, int]] = set()
        self.blockage_target_ratio_history: List[float] = []
        self.blockage_current_ratio_stochastic_history: List[float] = []
        self.blockage_gap_history: List[float] = []
        self.blockage_global_gate_history: List[float] = []
        self.newly_blocked_edge_count_history: List[int] = []
        self.blocked_edge_count_stochastic_history: List[int] = []
        self.blocked_edge_count_forced_island_history: List[int] = []
        self.blocked_edge_count_total_history: List[int] = []
        edge_lens: List[float] = []
        for src, nbs in self.topo.adjacency.items():
            for dst in nbs:
                if src >= dst:
                    continue
                edge_lens.append(float(self.topo.edge_distance(int(src), int(dst))))
        if edge_lens:
            self.edge_len_ref_m = float(np.percentile(np.asarray(edge_lens), 90))
        else:
            self.edge_len_ref_m = 1.0
        self.edge_len_ref_m = float(max(self.edge_len_ref_m, 1e-6))
        self.total_edge_count = int(max(sum(len(v) for v in self.topo.adjacency.values()) // 2, 1))
        self._init_field()

    def set_forced_island_edges(self, edges: set[Tuple[int, int]]) -> None:
        self.forced_island_edge_keys = {
            (int(min(a, b)), int(max(a, b))) for a, b in set(edges or set())
        }
        # Keep legacy alias synchronized so older callers that still think in terms
        # of non-reopen edges continue to work.
        self.nonreopen_edge_keys = set(self.forced_island_edge_keys)
        self._refresh_blockage_partition_stats()

    def set_nonreopen_edges(self, edges: set[Tuple[int, int]]) -> None:
        self.set_forced_island_edges(edges)

    def _blockage_profile(self) -> Tuple[float, float]:
        sc = str(getattr(self.cfg, "scenario", "B")).upper().strip()
        if sc == "B":
            base_asymptote = float(getattr(self.cfg, "blockage_asymptote_B", 0.12))
        elif sc == "C":
            base_asymptote = float(getattr(self.cfg, "blockage_asymptote_C", 0.30))
        else:
            base_asymptote = 0.0

        cx = str(getattr(self.cfg, "map_complexity", "M")).upper().strip()
        if cx == "L":
            scale = float(getattr(self.cfg, "blockage_asymptote_scale_L", 0.80))
            tau = float(getattr(self.cfg, "blockage_tau_steps_L", 150.0))
        elif cx == "R":
            scale = float(getattr(self.cfg, "blockage_asymptote_scale_R", 0.40))
            tau = float(getattr(self.cfg, "blockage_tau_steps_R", 140.0))
        else:
            scale = float(getattr(self.cfg, "blockage_asymptote_scale_M", 1.00))
            tau = float(getattr(self.cfg, "blockage_tau_steps_M", 110.0))

        return float(np.clip(base_asymptote * scale, 0.0, 1.0)), float(max(tau, 1.0))

    def _target_block_ratio(self, step_index: int) -> float:
        b_inf, tau = self._blockage_profile()
        self.blockage_curve_B_inf = float(b_inf)
        self.blockage_curve_tau_steps = float(tau)
        if b_inf <= 0.0:
            return 0.0
        t = float(max(step_index, 0))
        return float(b_inf * (1.0 - np.exp(-t / max(tau, 1e-6))))

    def _refresh_blockage_partition_stats(self) -> None:
        final_blocked = set(self.topo.blocked_edges)
        forced_blocked = set(final_blocked & self.forced_island_edge_keys)
        stochastic_blocked = set(final_blocked - forced_blocked)
        self.stochastic_blocked_edge_keys = set(stochastic_blocked)
        total = float(max(self.total_edge_count, 1))
        self.blocked_edge_count_stochastic = int(len(stochastic_blocked))
        self.blocked_edge_count_forced_island = int(len(forced_blocked))
        self.blocked_edge_count_total = int(len(final_blocked))
        self.blocked_edge_count_stochastic_step = int(self.blocked_edge_count_stochastic)
        self.blocked_edge_count_forced_island_step = int(self.blocked_edge_count_forced_island)
        self.blocked_edge_count_total_step = int(self.blocked_edge_count_total)
        self.blocked_edge_count = int(self.blocked_edge_count_total)
        self.blocked_ratio_stochastic = float(self.blocked_edge_count_stochastic / total)
        self.blocked_ratio_forced_island = float(self.blocked_edge_count_forced_island / total)
        self.blocked_ratio_total = float(self.blocked_edge_count_total / total)

    def _init_field(self) -> None:
        self.epicenter_node = self._closest_node_id(
            float(self.field.quake_epicenter[0]),
            float(self.field.quake_epicenter[1]),
        )
        for node_id, node in self.topo.nodes.items():
            self.node_hazard[node_id] = self.weather_at((node.x, node.y))

    def _closest_node_id(self, x: float, y: float) -> int:
        best_id = 0
        best_d = float("inf")
        for nid, node in self.topo.nodes.items():
            d = float(np.hypot(float(node.x) - x, float(node.y) - y))
            if d < best_d:
                best_d = d
                best_id = int(nid)
        return best_id

    def _point_cache_key(self, x: float, y: float, *, resolution_m: float = 1.0) -> Tuple[int, int]:
        res = float(max(resolution_m, 1e-6))
        return (int(round(float(x) / res)), int(round(float(y) / res)))

    def _ensure_weather_caches(self) -> None:
        cur = int(self.step_index)
        if self._weather_cache_step != cur:
            self._weather_cache_step = cur
            self._weather_cache.clear()
        if self._field_hazard_cache_step != cur:
            self._field_hazard_cache_step = cur
            self._field_hazard_cache.clear()

    def _field_hazard_at_cached(self, x: float, y: float) -> Tuple[float, float, float, float, float]:
        self._ensure_weather_caches()
        key = self._point_cache_key(x, y, resolution_m=1.0)
        cached = self._field_hazard_cache.get(key, None)
        if cached is not None:
            return cached
        out = self.field.get_hazard_at(float(x), float(y))
        self._field_hazard_cache[key] = out
        return out

    def _nearest_node_for_xy_cached(self, x: float, y: float) -> Optional[int]:
        key = self._point_cache_key(x, y, resolution_m=5.0)
        cached = self._nearest_node_xy_cache.get(key, None)
        if cached is not None:
            return int(cached)
        nearest_nid = None
        nearest_d = float('inf')
        for nid, node in self.topo.nodes.items():
            d = float(np.hypot(float(node.x) - x, float(node.y) - y))
            if d < nearest_d:
                nearest_d = d
                nearest_nid = int(nid)
        if nearest_nid is not None:
            self._nearest_node_xy_cache[key] = int(nearest_nid)
        return nearest_nid

    def wind_vector_at(
        self,
        point_xy: Tuple[float, float],
        base_wind_vector: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float]:
        _, _, vx, vy, _ = self._field_hazard_at_cached(float(point_xy[0]), float(point_xy[1]))
        # Keep backward signature; optional base wind vector is treated as additive bias.
        if base_wind_vector is not None:
            vx += float(base_wind_vector[0])
            vy += float(base_wind_vector[1])
        return float(vx), float(vy)

    def rainfall_at(self, point_xy: Tuple[float, float]) -> float:
        _, _, _, _, rain = self._field_hazard_at_cached(float(point_xy[0]), float(point_xy[1]))
        return float(rain)

    def weather_at(self, point_xy: Tuple[float, float]) -> NodeHazard:
        x = float(point_xy[0])
        y = float(point_xy[1])
        self._ensure_weather_caches()
        cache_key = self._point_cache_key(x, y, resolution_m=1.0)
        cached = self._weather_cache.get(cache_key, None)
        if cached is not None:
            return cached
        quake, wind, _, _, rain = self._field_hazard_at_cached(x, y)
        if self.real_case_enabled and self.earthquake_field_mode == "usgs_shakemap":
            nearest_nid = self._nearest_node_for_xy_cached(x, y)
            if nearest_nid is not None:
                quake = float(
                    np.clip(
                        getattr(self.topo.nodes[int(nearest_nid)], "quake_norm", 0.0),
                        0.0,
                        1.0,
                    )
                )
        if (
            self.real_case_enabled
            and self.earthquake_field_mode == "legacy_proxy"
            and self.real_case_hazard_profile == "wenchuan_frontline_v1"
        ):
            xs = np.array([n.x for n in self.topo.nodes.values()], dtype=np.float64)
            ys = np.array([n.y for n in self.topo.nodes.values()], dtype=np.float64)
            x_min = float(xs.min()) if xs.size else 0.0
            x_max = float(xs.max()) if xs.size else 1.0
            y_min = float(ys.min()) if ys.size else 0.0
            y_max = float(ys.max()) if ys.size else 1.0
            sx = float(max(x_max - x_min, 1.0))
            sy = float(max(y_max - y_min, 1.0))
            xn = float(np.clip((x - x_min) / sx, 0.0, 1.0))
            yn = float(np.clip((y - y_min) / sy, 0.0, 1.0))
            mountain_front = float(np.clip(0.68 * (1.0 - xn) + 0.32 * yn, 0.0, 1.0))
            river_corridor = float(np.exp(-((x - float(self._river_corridor_x)) ** 2) / (2.0 * max((0.10 * sx) ** 2, 1e-6))))
            nearest_nid = self._nearest_node_for_xy_cached(x, y)
            builtup = float(getattr(self.topo.nodes.get(int(nearest_nid), None), "builtup_intensity", 0.0)) if nearest_nid is not None else 0.0
            open_exposure = float(np.clip(0.75 * (1.0 - builtup) + 0.25 * river_corridor, 0.0, 1.0))
            quake = float(np.clip(quake * (0.70 + 0.75 * mountain_front), 0.0, 1.0))
            rain = float(max(rain * (0.55 + 0.95 * river_corridor), 0.0))
            wind = float(max(wind * (0.65 + 0.60 * open_exposure), 0.0))
        out = NodeHazard(
            rain=float(max(rain, 0.0)),
            wind=float(max(wind, 0.0)),
            quake=float(np.clip(quake, 0.0, 1.0)),
        )
        self._weather_cache[cache_key] = out
        return out

    def step(self) -> Tuple[float, float, float, int]:
        self.step_index += 1
        self._weather_cache_step = -1
        self._field_hazard_cache_step = -1
        self._weather_cache.clear()
        self._field_hazard_cache.clear()
        self.field.step()
        self.epicenter_node = self._closest_node_id(
            float(self.field.quake_epicenter[0]),
            float(self.field.quake_epicenter[1]),
        )

        for node_id, node in self.topo.nodes.items():
            self.node_hazard[node_id] = self.weather_at((node.x, node.y))

        self._refresh_blockage_partition_stats()
        current_ratio_stochastic = float(self.blocked_ratio_stochastic)
        # The published 0.25 curve is the TOTAL blocked-road target. Forced
        # island cuts are part of that budget, not an extra percentage added on
        # top of it.
        target_ratio = float(self._target_block_ratio(self.step_index))
        target_ratio_stochastic = float(
            max(target_ratio - float(self.blocked_ratio_forced_island), 0.0)
        )
        gap = float(max(target_ratio_stochastic - current_ratio_stochastic, 0.0))
        alive_ratio = float(max(1.0 - current_ratio_stochastic, 1e-6))
        gain_k = float(getattr(self.cfg, "blockage_curve_gain_k", 1.0)) if self.cfg is not None else 1.0
        gate_cap = float(getattr(self.cfg, "blockage_curve_gate_cap", 0.12)) if self.cfg is not None else 0.12
        if bool(getattr(self.cfg, "blockage_curve_enabled", True)):
            global_gate = float(np.clip(gain_k * gap / alive_ratio, 0.0, gate_cap))
            phase = "curve_v2"
        else:
            global_gate = float(np.clip(float(getattr(self.cfg, "lambda_aggressive", 0.0)), 0.0, gate_cap))
            phase = "legacy_constant"

        self.blockage_target_ratio_step = float(target_ratio)
        self.blockage_target_ratio_stochastic_step = float(target_ratio_stochastic)
        self.blockage_current_ratio_stochastic_step = float(current_ratio_stochastic)
        self.blockage_gap_step = float(gap)
        self.blockage_global_gate_step = float(global_gate)
        self.last_percolation_phase = phase
        self.last_lambda = float(global_gate)

        prev_blocked_total = set(self.topo.blocked_edges)

        pmacro_vals: List[float] = []
        pstep_vals: List[float] = []
        bldg_vals: List[float] = []
        infra_vals: List[float] = []
        length_vals: List[float] = []
        self.last_edge_pstep = {}
        sampled_block_candidates: List[Tuple[float, int, int]] = []
        for src, nbs in self.topo.adjacency.items():
            for dst in nbs:
                if src >= dst:
                    continue
                edge_key = (int(min(src, dst)), int(max(src, dst)))
                if edge_key in prev_blocked_total or edge_key in self.forced_island_edge_keys:
                    continue

                hs, hd = self.node_hazard[int(src)], self.node_hazard[int(dst)]
                eattr = self.topo.edge_attr(src, dst)
                slope = float(
                    np.clip(
                        0.5
                        * (
                            self.topo.nodes[src].slope_norm + self.topo.nodes[dst].slope_norm
                        ),
                        0.0,
                        1.0,
                    )
                )
                rain = float(
                    np.clip(
                        0.5 * (hs.rain + hd.rain) / max(self.base_rainfall_mmh, 1e-6),
                        0.0,
                        2.5,
                    )
                )
                quake = float(np.clip(0.5 * (hs.quake + hd.quake), 0.0, 2.5))
                v_base = float(np.clip(eattr.base_vulnerability, 0.0, 1.0))
                bldg = float(np.clip(eattr.building_density_norm, 0.0, 1.0))
                infra = float(np.clip(eattr.infra_bottleneck_norm, 0.0, 1.0))
                edge_len = float(self.topo.edge_distance(int(src), int(dst)))
                len_norm = float(np.clip(edge_len / max(self.edge_len_ref_m, 1e-6), 0.0, 2.5))

                if self.cfg is None:
                    b0, bs, br, be, bsr, bre, bv = -4.0, 1.4, 1.2, 1.8, 3.0, 1.1, 1.0
                    bb, bi, bl, blr = 0.9, 1.3, 0.8, 0.9
                    pmax = 0.85
                else:
                    b0 = float(self.cfg.logistic_beta0)
                    bs = float(self.cfg.logistic_beta_slope)
                    br = float(self.cfg.logistic_beta_rain)
                    be = float(self.cfg.logistic_beta_quake)
                    bsr = float(self.cfg.logistic_beta_sr)
                    bre = float(self.cfg.logistic_beta_re)
                    bv = float(self.cfg.logistic_beta_vbase)
                    bb = float(self.cfg.logistic_beta_bldg)
                    bi = float(self.cfg.logistic_beta_infra)
                    bl = float(self.cfg.logistic_beta_length)
                    blr = float(self.cfg.logistic_beta_lr)
                    pmax = float(self.cfg.stochastic_block_max_prob)

                if self.real_case_enabled and self.rb_road_damage_mode == "earthquake_only":
                    # Weather remains active for UAV operation; it is excluded
                    # only from the earthquake-related road disruption score.
                    br = bsr = bre = blr = 0.0

                z = (
                    b0
                    + bs * slope
                    + br * rain
                    + be * quake
                    + bsr * (slope * rain)
                    + bre * (rain * quake)
                    + bb * bldg
                    + bi * infra
                    + bl * len_norm
                    + blr * (len_norm * rain)
                    + bv * v_base
                )
                p_macro = float(1.0 / (1.0 + np.exp(-z)))
                p_step = float(np.clip(p_macro * global_gate, 0.0, pmax))

                self.last_edge_pstep[edge_key] = p_step
                pmacro_vals.append(p_macro)
                pstep_vals.append(p_step)
                bldg_vals.append(bldg)
                infra_vals.append(infra)
                length_vals.append(len_norm)

                draw = float(self.rng.uniform())
                if draw < p_step:
                    # Rank successful draws by normalized margin, then apply a
                    # hard count cap below. This preserves vulnerability bias
                    # without allowing one step to overshoot the total curve.
                    sampled_block_candidates.append(
                        (float(draw / max(p_step, 1e-12)), int(src), int(dst))
                    )

        stochastic_target_edge_count = int(
            max(np.floor(target_ratio_stochastic * float(self.total_edge_count) + 1e-9), 0)
        )
        stochastic_slots = int(
            max(stochastic_target_edge_count - len(self.stochastic_blocked_edge_keys), 0)
        )
        sampled_block_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        for _score, src, dst in sampled_block_candidates[:stochastic_slots]:
            self.topo.set_blocked(int(src), int(dst), True)

        final_blocked = set(self.topo.blocked_edges)
        new_age: Dict[Tuple[int, int], int] = {}
        for edge_key in final_blocked:
            if edge_key in prev_blocked_total:
                new_age[edge_key] = int(self.blocked_edge_age_steps.get(edge_key, 0)) + 1
            else:
                new_age[edge_key] = 0
        self.blocked_edge_age_steps = new_age

        newly_stochastic = set((final_blocked - prev_blocked_total) - self.forced_island_edge_keys)
        self.newly_blocked_edge_count_step = int(len(newly_stochastic))
        self.newly_blocked_edge_count_total += int(self.newly_blocked_edge_count_step)

        self._refresh_blockage_partition_stats()
        self.ever_blocked_edge_keys.update(self.stochastic_blocked_edge_keys)

        self.last_pmacro_mean = float(np.mean(pmacro_vals)) if pmacro_vals else 0.0
        self.last_pstep_mean = float(np.mean(pstep_vals)) if pstep_vals else 0.0
        self.last_bldg_mean = float(np.mean(bldg_vals)) if bldg_vals else 0.0
        self.last_infra_mean = float(np.mean(infra_vals)) if infra_vals else 0.0
        self.last_length_norm_mean = float(np.mean(length_vals)) if length_vals else 0.0

        self.blockage_target_ratio_history.append(float(self.blockage_target_ratio_step))
        self.blockage_current_ratio_stochastic_history.append(float(self.blockage_current_ratio_stochastic_step))
        self.blockage_gap_history.append(float(self.blockage_gap_step))
        self.blockage_global_gate_history.append(float(self.blockage_global_gate_step))
        self.newly_blocked_edge_count_history.append(int(self.newly_blocked_edge_count_step))
        self.blocked_edge_count_stochastic_history.append(int(self.blocked_edge_count_stochastic_step))
        self.blocked_edge_count_forced_island_history.append(int(self.blocked_edge_count_forced_island_step))
        self.blocked_edge_count_total_history.append(int(self.blocked_edge_count_total_step))

        rains = [h.rain for h in self.node_hazard.values()]
        winds = [h.wind for h in self.node_hazard.values()]
        return (
            float(np.mean(rains)) if rains else 0.0,
            float(np.mean(winds)) if winds else 0.0,
            float(self.blocked_ratio_total),
            int(self.epicenter_node),
        )

    def node_weather(self, node_id: int) -> NodeHazard:
        return self.node_hazard[int(node_id)]


