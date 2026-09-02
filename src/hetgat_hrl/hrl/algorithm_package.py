from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hetgat_hrl.core.algorithm_profile import AlgorithmProfile
from hetgat_hrl.core.mdp_spec import EnvConfig


@dataclass(frozen=True)
class AlgorithmPackage:
    """A planner bundled with its private runtime settings and identity."""

    algorithm_id: str
    planner: Any
    planner_class: str
    backend_family: str
    encoder_type: str
    enable_rth_mask: bool
    runtime_cfg: EnvConfig
    profile: AlgorithmProfile
    public_environment_hash: str
    algorithm_config_hash: str

    def bind_environment(self, env: Any) -> None:
        env.algorithm_profile = self.profile
        env.current_method = str(self.algorithm_id)
