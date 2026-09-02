from __future__ import annotations

from hetgat_hrl.hrl.event_responsive_alns_planner import EventResponsiveALNSPlanner


class RollingHorizonALNSPlanner(EventResponsiveALNSPlanner):
    """Fixed-interval rolling-horizon ALNS baseline."""

    algorithm_id = "rolling_horizon_alns"
