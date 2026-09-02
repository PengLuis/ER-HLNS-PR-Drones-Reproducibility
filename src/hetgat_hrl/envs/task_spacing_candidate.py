"""Candidate-only task sampler for the LB task-spacing sensitivity screen.

This module deliberately lives outside :mod:`base_env`.  The canonical
environment keeps its historical task sampler and therefore keeps the frozen
physical/task contract intact.  ``TaskSpacingDisasterEnv`` only changes the
seed-owned placement of synthetic-map tasks:

* emergency nodes are replayed exactly from the base environment's
  epicentre-biased seed draw and used as spacing anchors;
* normal nodes are then selected/repaired from the remaining,
  depot-reachable nodes;
* a 1,600 m cross-class separation is attempted before a transparent
  farthest-point fallback is used when the graph cannot supply eight such
  nodes.

No roads, hazard values, deadlines, demands, or task counts are changed by
the sampler.  In particular, it never blocks an edge and never creates an
island.  Any persistent islands observed in a run therefore come only from
the caller's existing environment configuration (for example, the historical
``forced_island_emergency_tasks`` setting), not from this candidate.
"""

from __future__ import annotations

from math import hypot, isfinite
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hetgat_hrl.envs.base_env import BaseHeteroDisasterEnv


class TaskSpacingDisasterEnv(BaseHeteroDisasterEnv):
    """Candidate environment with emergency-anchored, separated placement.

    The class is intended for sensitivity experiments only.  It is safe to
    instantiate with the canonical ``EnvConfig`` because the base reset still
    owns topology, hazard, deadline, demand, and road-event construction.
    """

    DEFAULT_MIN_CROSS_CLASS_SPACING_M = 1600.0

    def __init__(
        self,
        cfg: Any = None,
        algorithm_profile: Any = None,
        *,
        min_cross_class_spacing_m: float = DEFAULT_MIN_CROSS_CLASS_SPACING_M,
    ) -> None:
        # ``BaseHeteroDisasterEnv.__init__`` builds the first state and invokes
        # our overridden sampler, so initialize diagnostics before super().
        self.min_cross_class_spacing_m = float(max(min_cross_class_spacing_m, 0.0))
        self.task_spacing_candidate_diagnostics: Dict[str, Any] = {}
        super().__init__(cfg, algorithm_profile=algorithm_profile)

    # Base's synthetic-realism gate intentionally follows a configuration
    # switch.  The candidate is explicit and should be callable with the
    # frozen LB configuration (where that switch is false), while fixed real
    # city task manifests must remain under the base real-case path.
    def _synthetic_realism_task_sampling_enabled(self) -> bool:
        if self._real_case_task_sampling_enabled():
            return False
        return True

    def _spacing_distance_m(self, node_a: int, node_b: int) -> Tuple[float, str]:
        """Return a reproducible Euclidean/road proxy distance.

        Synthetic L nodes always have finite coordinates, so Euclidean distance
        is the primary metric.  A shortest-path distance is retained as a
        defensive fallback for custom topologies whose coordinates are not
        finite.  Blocked edges are ignored deliberately: this is task-space
        separation and must not depend on a sampled road realization.
        """

        try:
            a = self.topology.nodes[int(node_a)]
            b = self.topology.nodes[int(node_b)]
            euclid = float(hypot(float(a.x) - float(b.x), float(a.y) - float(b.y)))
        except (KeyError, TypeError, ValueError):
            euclid = float("nan")
        if isfinite(euclid):
            return euclid, "euclidean"
        try:
            road = float(self.topology.shortest_path_distance(int(node_a), int(node_b), ignore_blocked=True))
        except Exception:
            road = float("inf")
        return road, "road_proxy"

    def _candidate_nodes(self) -> Tuple[List[int], int]:
        """Return unique nodes that are connected to the depot in the graph."""

        all_nodes = [
            int(nid)
            for nid in self._task_candidate_node_ids(include_depot=False).tolist()
            if int(nid) in self.topology.nodes and int(nid) != 0
        ]
        reachable: List[int] = []
        unreachable = 0
        for nid in all_nodes:
            try:
                ok = bool(self.topology.path_exists(0, int(nid), ignore_blocked=True))
            except Exception:
                try:
                    ok = bool(isfinite(self.topology.shortest_path_distance(0, int(nid), ignore_blocked=True)))
                except Exception:
                    ok = False
            if ok:
                reachable.append(int(nid))
            else:
                unreachable += 1
        # ``sorted(dict.fromkeys(...))`` is intentional: node ordering must not
        # inherit hash iteration order, while the RNG still controls tie breaks.
        return sorted(dict.fromkeys(reachable)), int(unreachable)

    def _min_cross_distance(self, normal: Sequence[int], emergency: Sequence[int]) -> float:
        distances: List[float] = []
        for nid in normal:
            for eid in emergency:
                value, _metric = self._spacing_distance_m(int(nid), int(eid))
                if isfinite(value):
                    distances.append(float(value))
        return float(min(distances)) if distances else float("inf")

    def _farthest_point_select(
        self,
        candidates: Sequence[int],
        count: int,
        anchors: Sequence[int],
        *,
        minimum_anchor_distance_m: Optional[float] = None,
    ) -> Tuple[List[int], List[int]]:
        """Select points by max-min distance, with deterministic RNG ties.

        ``anchors`` are emergency nodes for the first pass and become the
        coverage anchors alongside already-selected normal nodes.  The return
        value contains selected nodes and the candidates rejected by the
        optional minimum-anchor-distance gate.
        """

        count = int(max(count, 0))
        if count <= 0:
            return [], []
        pool = [int(n) for n in dict.fromkeys(int(x) for x in candidates)]
        anchor_list = [int(x) for x in dict.fromkeys(int(x) for x in anchors)]
        spacing = None if minimum_anchor_distance_m is None else float(max(minimum_anchor_distance_m, 0.0))

        rejected_by_spacing: List[int] = []
        if spacing is not None and anchor_list:
            strict_pool: List[int] = []
            for nid in pool:
                dvals = [self._spacing_distance_m(int(nid), int(a))[0] for a in anchor_list]
                if dvals and all(isfinite(d) and float(d) + 1e-9 >= spacing for d in dvals):
                    strict_pool.append(int(nid))
                else:
                    rejected_by_spacing.append(int(nid))
            pool = strict_pool

        # Shuffle once to give equal-score candidates seed-owned tie breaks.
        # Python's stable max() then makes the tie break reproducible.
        self.rng.shuffle(pool)
        selected: List[int] = []
        current_anchors = list(anchor_list)
        while pool and len(selected) < count:
            def score(nid: int) -> Tuple[float, float, int]:
                distances = [
                    float(self._spacing_distance_m(int(nid), int(other))[0])
                    for other in current_anchors
                    if int(other) != int(nid)
                ]
                finite = [d for d in distances if isfinite(d)]
                min_dist = min(finite) if finite else float("inf")
                # A small deterministic secondary term prefers points far from
                # the depot only when coverage scores are exactly tied.
                depot_d, _ = self._spacing_distance_m(0, int(nid))
                return float(min_dist), float(depot_d if isfinite(depot_d) else -1.0), -int(nid)

            best = max(pool, key=score)
            selected.append(int(best))
            current_anchors.append(int(best))
            pool.remove(int(best))
        return selected, rejected_by_spacing

    def _sample_synthetic_realism_task_nodes(self, normal_count: int, emergency_count: int) -> Tuple[List[int], List[int]]:
        """Sample emergency tasks first, then separated normal coverage points."""

        normal_count = int(max(normal_count, 0))
        emergency_count = int(max(emergency_count, 0))
        candidates, unreachable_count = self._candidate_nodes()
        if len(candidates) < normal_count + emergency_count:
            raise ValueError(
                "Task-spacing candidate cannot satisfy unique reachable task count: "
                f"requested={normal_count + emergency_count}, reachable={len(candidates)}"
            )

        # Preserve the formal emergency realization for the same seed.  The
        # frozen sampler consumes RNG in the order ``normal uniform`` followed
        # by ``emergency epicentre-biased``.  Replaying those two draws before
        # replacing only the normal nodes makes the comparison a paired task
        # layout sensitivity test: TC locations, hazard realization and all
        # later RNG-dependent state remain unchanged, while routine locations
        # are redesigned.  We do not let an emergency-first redesign silently
        # trade away emergency performance.
        formal_candidates = [
            int(nid)
            for nid in self._task_candidate_node_ids(include_depot=False).tolist()
        ]
        formal_normal = self._sample_uniform_task_nodes(
            normal_count,
            candidate_node_ids=formal_candidates,
            replace=False,
        )
        formal_normal_set = set(int(x) for x in formal_normal)
        formal_remaining = [int(n) for n in formal_candidates if int(n) not in formal_normal_set]
        emergency_nodes = self._sample_emergency_task_nodes(
            emergency_count,
            candidate_node_ids=formal_remaining,
            replace=False,
        )
        self.task_spacing_candidate_diagnostics["emergency_sampling_preserved_formal_seed"] = True
        emergency_nodes = list(dict.fromkeys(int(x) for x in emergency_nodes))
        if len(emergency_nodes) < emergency_count:
            remaining = [int(x) for x in candidates if int(x) not in set(emergency_nodes)]
            self.rng.shuffle(remaining)
            emergency_nodes.extend(remaining[: emergency_count - len(emergency_nodes)])
        emergency_nodes = emergency_nodes[:emergency_count]

        remaining = [int(x) for x in candidates if int(x) not in set(emergency_nodes)]

        # Repair only the formal normal nodes that violate the cross-class
        # spacing rule.  This preserves the paired seed's emergency nodes and
        # keeps the ordinary-task depot/travel distribution close to the
        # formal draw; selecting globally farthest points would create an
        # artificial long-haul burden and confound spacing with difficulty.
        strict_normals: List[int] = []
        repair_targets: List[int] = []
        for nid in [int(x) for x in formal_normal[:normal_count]]:
            if int(nid) in set(remaining) and all(
                self._spacing_distance_m(int(nid), int(eid))[0] + 1e-9
                >= float(self.min_cross_class_spacing_m)
                for eid in emergency_nodes
            ):
                strict_normals.append(int(nid))
            else:
                repair_targets.append(int(nid))

        # The pool is shuffled only for deterministic RNG-owned tie breaking;
        # the primary score minimizes the change in depot shortest-path ETA,
        # then geometric displacement from the replaced node.
        remaining_pool = [int(x) for x in remaining if int(x) not in set(strict_normals)]
        self.rng.shuffle(remaining_pool)
        rejected: List[int] = []
        unfilled_targets: List[int] = []
        for target in repair_targets:
            target_eta = float(self.topology.shortest_path_distance(0, int(target), ignore_blocked=True))
            feasible: List[Tuple[float, float, float, int]] = []
            for nid in remaining_pool:
                if not all(
                    self._spacing_distance_m(int(nid), int(eid))[0] + 1e-9
                    >= float(self.min_cross_class_spacing_m)
                    for eid in emergency_nodes
                ):
                    continue
                eta = float(self.topology.shortest_path_distance(0, int(nid), ignore_blocked=True))
                disp = float(self._spacing_distance_m(int(nid), int(target))[0])
                feasible.append((abs(eta - target_eta), disp, eta, int(nid)))
            if feasible:
                feasible.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
                chosen = int(feasible[0][3])
                strict_normals.append(chosen)
                remaining_pool.remove(chosen)
            else:
                rejected.extend(int(x) for x in remaining_pool)
                unfilled_targets.append(int(target))

        fallback_used = len(strict_normals) < normal_count
        fallback_count = max(normal_count - len(strict_normals), 0)
        if fallback_used:
            # If the requested separation is infeasible, use the same ETA-aware
            # replacement without the hard gate.  The diagnostic records this
            # explicitly; the candidate never creates duplicates/islands.
            for target in unfilled_targets:
                if len(strict_normals) >= normal_count or not remaining_pool:
                    break
                target_eta = float(self.topology.shortest_path_distance(0, int(target), ignore_blocked=True))
                remaining_pool.sort(
                    key=lambda nid: (
                        abs(float(self.topology.shortest_path_distance(0, int(nid), ignore_blocked=True)) - target_eta),
                        int(nid),
                    )
                )
                strict_normals.append(int(remaining_pool.pop(0)))

        if len(strict_normals) != normal_count:
            raise ValueError(
                "Task-spacing candidate could not select the requested unique normal tasks: "
                f"requested={normal_count}, selected={len(strict_normals)}"
            )

        normal_nodes = [int(x) for x in strict_normals]
        all_selected = normal_nodes + [int(x) for x in emergency_nodes]
        if len(set(all_selected)) != len(all_selected):
            raise AssertionError("task-spacing candidate selected duplicate task nodes")
        if any(int(x) not in set(candidates) for x in all_selected):
            raise AssertionError("task-spacing candidate selected an unreachable task node")

        metric_counts: Dict[str, int] = {}
        for nid in normal_nodes:
            for eid in emergency_nodes:
                _dist, metric = self._spacing_distance_m(int(nid), int(eid))
                metric_counts[metric] = int(metric_counts.get(metric, 0)) + 1
        actual_min = self._min_cross_distance(normal_nodes, emergency_nodes)
        strict_selected_count = 0
        for nid in normal_nodes:
            if all(
                self._spacing_distance_m(int(nid), int(eid))[0] + 1e-9
                >= float(self.min_cross_class_spacing_m)
                for eid in emergency_nodes
            ):
                strict_selected_count += 1
        self.task_spacing_candidate_diagnostics = {
            "candidate_name": "lb_task_spacing_emergency_first_v1",
            "emergency_first": True,
            "emergency_sampling_preserved_formal_seed": True,
            "requested_normal_count": int(normal_count),
            "requested_emergency_count": int(emergency_count),
            "selected_normal_count": int(len(normal_nodes)),
            "selected_emergency_count": int(len(emergency_nodes)),
            "candidate_node_count": int(len(candidates) + unreachable_count),
            "reachable_candidate_node_count": int(len(candidates)),
            "unreachable_candidate_node_count": int(unreachable_count),
            "requested_min_cross_class_spacing_m": float(self.min_cross_class_spacing_m),
            "actual_min_cross_class_spacing_m": float(actual_min),
            "strict_normal_candidate_count": int(len(remaining) - len(rejected)),
            "strict_normal_selected_count": int(strict_selected_count),
            "spacing_fallback_used": bool(fallback_used),
            "spacing_fallback_count": int(fallback_count),
            "spacing_fallback_rejected_candidate_count": int(len(rejected)),
            "distance_metric_counts": metric_counts,
            "sampler_created_forced_island": False,
            "road_process_mutated_by_sampler": False,
        }
        return normal_nodes, emergency_nodes


__all__ = ["TaskSpacingDisasterEnv"]
