"""RB-v3 difficulty-calibration adapter layered on the frozen RB-v2 physics."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from hetgat_hrl.core.algorithm_profile import AlgorithmProfile
from hetgat_hrl.core.mdp_spec import EnvConfig, JointState
from hetgat_hrl.envs.rb_v2_environment import RBV2HeteroDisasterEnv


class RBV3HeteroDisasterEnv(RBV2HeteroDisasterEnv):
    """Use manifest deadlines while retaining RB-v2 hourly weather semantics."""

    def __init__(
        self,
        cfg: EnvConfig,
        *,
        weather_profile_path: Path,
        weather_repeat: bool = False,
        algorithm_profile: Optional[AlgorithmProfile] = None,
    ) -> None:
        super().__init__(
            cfg,
            weather_profile_path=weather_profile_path,
            deadline_policy="manifest",
            weather_repeat=weather_repeat,
            algorithm_profile=algorithm_profile,
        )
        self._restore_v3_forced_islands()

    def _restore_v3_forced_islands(self) -> None:
        """Re-register the selected incident edges after base runtime init."""
        forced_count = int(max(getattr(self.cfg, "forced_island_emergency_tasks", 0), 0))
        if forced_count <= 0:
            return
        critical = [
            task
            for task in self.state.tasks.values()
            if task.task_class == "time_critical_lightweight"
        ]
        critical.sort(
            key=lambda task: (
                float(self.topology.shortest_path_distance(0, int(task.demand_node), ignore_blocked=True))
                / max(len(self.topology.adjacency.get(int(task.demand_node), set())), 1),
                str(task.task_id),
            ),
            reverse=True,
        )
        selected = critical[:forced_count]
        edges: set[tuple[int, int]] = set()
        for task in selected:
            node = int(task.demand_node)
            for neighbor in self.topology.adjacency.get(node, set()):
                edge = (min(node, int(neighbor)), max(node, int(neighbor)))
                edges.add(edge)
                self.topology.set_blocked(*edge, True)
        self._forced_island_candidate_task_ids = {str(task.task_id) for task in selected}
        self._forced_island_task_ids = set(self._forced_island_candidate_task_ids)
        self._forced_island_edge_keys = edges
        if hasattr(self.hazards, "set_forced_island_edges"):
            self.hazards.set_forced_island_edges(edges)

    def reset(self, seed: Optional[int] = None) -> JointState:
        state = super().reset(seed=seed)
        self._restore_v3_forced_islands()
        return state
