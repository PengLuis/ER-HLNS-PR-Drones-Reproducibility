"""RB-v2 environment adapter for a checked-in hourly weather replay.

The canonical v1 runtime is raw-SHA frozen.  This module deliberately wraps
that runtime instead of changing it, so old v1 artifacts remain executable.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from hetgat_hrl.core.algorithm_profile import AlgorithmProfile
from hetgat_hrl.core.mdp_spec import EnvConfig, JointState
from hetgat_hrl.envs.base_env import BaseHeteroDisasterEnv
from hetgat_hrl.envs.hazards import DynamicHazardField, NodeHazard


@dataclass(frozen=True)
class HourlyWeatherRecord:
    precipitation_mmh: float
    wind_speed_10m_mps: float
    wind_direction_10m_deg: float


class HourlyUniformWeatherHazard:
    """Delegate roads/quake to v1 while replacing only operational weather."""

    def __init__(
        self,
        base: DynamicHazardField,
        profile_path: Path,
        *,
        dt_seconds: float,
        repeat: bool = False,
    ) -> None:
        self._base = base
        self._dt_seconds = float(dt_seconds)
        self._repeat = bool(repeat)
        if self._dt_seconds <= 0.0:
            raise ValueError("dt_seconds must be positive")
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        if payload.get("schema") != "hourly_uniform_weather_profile_v1":
            raise ValueError("unexpected hourly weather profile schema")
        self.interval_seconds = float(payload.get("interval_seconds", 0.0))
        if self.interval_seconds <= 0.0:
            raise ValueError("weather profile interval_seconds must be positive")
        records = payload.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError("weather profile must contain records")
        parsed = []
        for index, row in enumerate(records):
            if not isinstance(row, Mapping):
                raise ValueError(f"weather record {index} must be an object")
            try:
                rain = float(row["precipitation_mmh"])
                speed = float(row["wind_speed_10m_mps"])
                direction = float(row["wind_direction_10m_deg"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid weather record {index}") from exc
            if not all(np.isfinite(value) for value in (rain, speed, direction)):
                raise ValueError(f"non-finite weather record {index}")
            if rain < 0.0 or speed < 0.0:
                raise ValueError(f"negative weather record {index}")
            parsed.append(HourlyWeatherRecord(rain, speed, direction % 360.0))
        self.records = tuple(parsed)
        self.step_index = int(base.step_index)
        self.node_hazard: dict[int, NodeHazard] = {}
        self._refresh_node_weather()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def _record(self) -> HourlyWeatherRecord:
        elapsed_seconds = float(max(self.step_index, 0)) * self._dt_seconds
        index = int(np.floor(elapsed_seconds / self.interval_seconds))
        if self._repeat:
            index %= len(self.records)
        else:
            index = min(index, len(self.records) - 1)
        return self.records[index]

    def _vector(self) -> Tuple[float, float]:
        row = self._record()
        theta = float(np.deg2rad(row.wind_direction_10m_deg))
        # Meteorological direction is FROM, clockwise from north.  Map axes are
        # x=east and y=north, so the motion vector points downwind.
        return (
            float(-row.wind_speed_10m_mps * np.sin(theta)),
            float(-row.wind_speed_10m_mps * np.cos(theta)),
        )

    def weather_at(self, point_xy: Tuple[float, float]) -> NodeHazard:
        row = self._record()
        quake = float(self._base.weather_at(point_xy).quake)
        return NodeHazard(
            rain=float(row.precipitation_mmh),
            wind=float(row.wind_speed_10m_mps),
            quake=quake,
        )

    def rainfall_at(self, point_xy: Tuple[float, float]) -> float:
        return float(self._record().precipitation_mmh)

    def wind_vector_at(
        self,
        point_xy: Tuple[float, float],
        base_wind_vector: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float]:
        vx, vy = self._vector()
        if base_wind_vector is not None:
            vx += float(base_wind_vector[0])
            vy += float(base_wind_vector[1])
        return float(vx), float(vy)

    def node_weather(self, node_id: int) -> NodeHazard:
        return self.node_hazard[int(node_id)]

    def _refresh_node_weather(self) -> None:
        self.node_hazard = {
            int(node_id): self.weather_at((float(node.x), float(node.y)))
            for node_id, node in self._base.topo.nodes.items()
        }

    def step(self) -> Tuple[float, float, float, int]:
        # The delegated v1 field advances quake and road disruption.  Its rain
        # is excluded from RB road ranking by earthquake_only; this wrapper then
        # publishes the checked-in operational weather for the new step.
        _, _, blocked_ratio, epicenter = self._base.step()
        self.step_index = int(self._base.step_index)
        self._refresh_node_weather()
        rain = float(self._record().precipitation_mmh)
        wind = float(self._record().wind_speed_10m_mps)
        return rain, wind, float(blocked_ratio), int(epicenter)


class RBV2HeteroDisasterEnv(BaseHeteroDisasterEnv):
    """Base v1 execution plus v2 RB weather and deadline overlays."""

    def __init__(
        self,
        cfg: EnvConfig,
        *,
        weather_profile_path: Path,
        deadline_policy: str = "config_template",
        weather_repeat: bool = False,
        algorithm_profile: Optional[AlgorithmProfile] = None,
    ) -> None:
        self._rb_v2_weather_profile_path = Path(weather_profile_path).resolve()
        self._rb_v2_deadline_policy = str(deadline_policy).strip().lower()
        self._rb_v2_weather_repeat = bool(weather_repeat)
        if self._rb_v2_deadline_policy not in {"manifest", "config_template"}:
            raise ValueError("deadline_policy must be manifest or config_template")
        super().__init__(cfg, algorithm_profile=algorithm_profile)
        self._install_rb_v2_overlays()

    def _install_rb_v2_overlays(self) -> None:
        self.hazards = HourlyUniformWeatherHazard(
            self.hazards,
            self._rb_v2_weather_profile_path,
            dt_seconds=self._dt_seconds,
            repeat=self._rb_v2_weather_repeat,
        )
        if self._rb_v2_deadline_policy == "config_template":
            routine = sorted(
                (task for task in self.state.tasks.values() if task.task_class == "routine_bulk"),
                key=lambda task: str(task.task_id),
            )
            critical = sorted(
                (task for task in self.state.tasks.values() if task.task_class == "time_critical_lightweight"),
                key=lambda task: str(task.task_id),
            )
            for index, task in enumerate(routine):
                task.deadline_step = int(
                    min(
                        self.cfg.max_steps - 1,
                        self.cfg.normal_task_deadline_start_step
                        + index * self.cfg.normal_task_deadline_interval_step,
                    )
                )
            for index, task in enumerate(critical):
                task.deadline_step = int(
                    min(
                        self.cfg.max_steps - 1,
                        self.cfg.emergency_task_deadline_start_step
                        + index * self.cfg.emergency_task_deadline_interval_step,
                    )
                )

    def reset(self, seed: Optional[int] = None) -> JointState:
        state = super().reset(seed=seed)
        self._install_rb_v2_overlays()
        return state
