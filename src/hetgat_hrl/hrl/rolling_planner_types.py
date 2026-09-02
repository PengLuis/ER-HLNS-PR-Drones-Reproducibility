from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class RollingPlannerState:
    step_last_refresh: int = 0
    goals: Dict[str, Optional[str]] = field(default_factory=dict)
    resolved_tasks_last: int = 0
    goal_assigned_step: Dict[str, int] = field(default_factory=dict)


@dataclass
class RollingPlannerWeights:
    # Truck task scoring
    truck_urgency: float = 1.20
    truck_eta: float = 1.00
    truck_risk: float = 0.70
    truck_demand: float = 0.60
    # UAV emergency scoring
    uav_urgency: float = 1.10
    uav_eta: float = 1.20
    uav_risk: float = 0.90
    uav_margin: float = 1.30
    uav_emergency_bonus: float = 0.40
    # UAV truck-rendezvous scoring
    uav_recovery_need: float = 1.50
    uav_truck_distance: float = 0.80
    # Shared keep-goal bonus
    keep_goal_bonus: float = 0.20
