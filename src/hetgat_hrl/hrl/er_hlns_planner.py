from __future__ import annotations

from hetgat_hrl.hrl.event_responsive_alns_planner import EventResponsiveALNSPlanner


class ERHLNSPlanner(EventResponsiveALNSPlanner):
    """Packaged paper mainline.

    Complete cooperative route planning is activated by the bound
    ``AlgorithmProfile`` rather than by a public environment switch.
    """

    algorithm_id = "er_hlns"
