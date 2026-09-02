from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class AlgorithmProfile:
    """Runtime identity and capabilities owned by one planning algorithm.

    Formal runners bind a profile explicitly so the public disaster environment
    never chooses a planner through a global experiment switch. Legacy callers
    without a profile retain the historical config-based behavior.
    """

    algorithm_id: str
    planner_class: str
    capabilities: Mapping[str, bool]

    def has(self, capability: str) -> bool:
        return bool(self.capabilities.get(str(capability), False))


ER_HLNS_ROUTE_PLAN_CAPABILITY = "er_hlns_route_plan"
# ER-HLNS-only coordination layer.  The capability covers the support,
# anchor, reservation and rescue helpers that live in the shared rolling
# planner implementation.  It is intentionally separate from the physical
# safety gates: comparator packages can disable the coordination helpers while
# retaining the same battery, payload, communication and recovery contracts.
ER_HLNS_COORDINATION_CAPABILITY = "er_hlns_coordination"
# Candidate-only upper-planning capability.  It allows a supporting truck to
# advance toward its next direct-routine stop while a docked UAV executes the
# current emergency sortie, but only after the route manager proves a safe
# recovery corridor.  It is deliberately separate from the formal ER-HLNS
# coordination capability so the frozen mainline remains unchanged.
ER_HLNS_PARALLEL_ROUTINE_EMERGENCY_CAPABILITY = (
    "er_hlns_parallel_routine_emergency"
)
# Candidate-only LB repair capability.  It permits one bounded takeover of a
# still-pending, never-started NORMAL task by an idle, stocked truck.  It is
# deliberately separate from the parallel-corridor pilot so formal ER-HLNS
# and the earlier candidate remain unchanged.
ER_HLNS_R4_ROUTINE_TAKEOVER_CAPABILITY = "er_hlns_r4_routine_takeover"
# Candidate-only upper-planning capability.  It lets an otherwise idle,
# stocked truck pick up one pending routine task when no emergency contract is
# inside its protected deadline reserve.  The formal ER-HLNS profile never
# exposes this capability.
ER_HLNS_IDLE_ROUTINE_DISPATCH_CAPABILITY = "er_hlns_idle_routine_dispatch"
# Candidate-only upper-planning capability for the aggressive LB pilot.  It
# deliberately remains separate from the existing idle-dispatch capability so
# the pilot can be audited/disabled without changing formal ER-HLNS behavior.
ER_HLNS_BALANCED_ALL_TASKS_CAPABILITY = "er_hlns_balanced_all_tasks"
ER_HLNS_BALANCED_ALL_TASKS_V2_CAPABILITY = "er_hlns_balanced_all_tasks_v2"
ER_HLNS_BALANCED_ALL_TASKS_V3_CAPABILITY = "er_hlns_balanced_all_tasks_v3"
UAV_SCOUT_INFORMATION_CAPABILITY = "uav_scout_information"
# Algorithm-owned coordination behavior: a loaded, docked UAV may retain its
# atomic emergency contract until launch. This is deliberately a capability,
# not a public environment parameter, so baselines share the same world.
ER_HLNS_B_PRELAUNCH_CONTRACT_LOCK_CAPABILITY = (
    "er_hlns_b_prelaunch_contract_lock"
)
ER_HLNS_B_DOCKED_LATCH_REARM_CAPABILITY = (
    "er_hlns_b_docked_latch_rearm"
)
# Candidate-only low-seed repair.  This is an algorithm capability rather
# than an environment parameter, so it cannot alter the frozen physical
# contract.  The route manager uses it to permit one narrowly guarded
# routine-task handoff while no emergency/UAV execution prefix is active.
ER_HLNS_LOW_SEED_RESCUE_CAPABILITY = "er_hlns_low_seed_rescue"


def er_hlns_route_plan_active(env: Any) -> bool:
    """Return whether the bound algorithm owns the ER-HLNS route planner."""

    profile = getattr(env, "algorithm_profile", None)
    if isinstance(profile, AlgorithmProfile):
        return profile.has(ER_HLNS_ROUTE_PLAN_CAPABILITY)
    cfg = getattr(env, "cfg", None)
    return bool(getattr(cfg, "hrl_route_plan_v2_enabled", False))


def uav_scout_information_active(env: Any) -> bool:
    """Return whether UAV road observations may enter shared awareness.

    Formal algorithms expose the information capability through their bound
    profile, keeping the physical sensor configuration common across methods.
    Legacy callers retain the historical config-based ablation behavior.
    """

    profile = getattr(env, "algorithm_profile", None)
    if isinstance(profile, AlgorithmProfile):
        return profile.has(UAV_SCOUT_INFORMATION_CAPABILITY)
    cfg = getattr(env, "cfg", None)
    return bool(getattr(cfg, "hrl_route_plan_uav_scout_enabled", True))


def er_hlns_coordination_active(env: Any) -> bool:
    """Return whether ER-HLNS support/anchor/rescue coordination is active.

    Bound algorithm packages own this capability.  Direct legacy callers that
    do not bind a profile retain the historical behavior (enabled), so this
    helper is backwards-compatible with existing planner/unit-test fixtures.
    """

    profile = getattr(env, "algorithm_profile", None)
    if isinstance(profile, AlgorithmProfile):
        return profile.has(ER_HLNS_COORDINATION_CAPABILITY)
    return True


def er_hlns_parallel_routine_emergency_active(env: Any) -> bool:
    """Return whether the candidate parallel routine/emergency pilot is active."""

    profile = getattr(env, "algorithm_profile", None)
    if isinstance(profile, AlgorithmProfile):
        return profile.has(ER_HLNS_PARALLEL_ROUTINE_EMERGENCY_CAPABILITY)
    # Direct legacy callers must opt in explicitly; this avoids silently
    # changing historical route-v2 behavior or the physical freeze.
    cfg = getattr(env, "cfg", None)
    return bool(
        getattr(cfg, "hrl_route_plan_parallel_routine_emergency_enabled", False)
    )


def er_hlns_r4_routine_takeover_active(env: Any) -> bool:
    """Return whether the candidate-only R4 routine takeover is active."""

    profile = getattr(env, "algorithm_profile", None)
    if isinstance(profile, AlgorithmProfile):
        return profile.has(ER_HLNS_R4_ROUTINE_TAKEOVER_CAPABILITY)
    cfg = getattr(env, "cfg", None)
    return bool(
        getattr(cfg, "hrl_route_plan_r4_routine_takeover_enabled", False)
    )


def er_hlns_idle_routine_dispatch_active(env: Any) -> bool:
    """Return whether candidate-only idle routine dispatch is active."""

    profile = getattr(env, "algorithm_profile", None)
    if isinstance(profile, AlgorithmProfile):
        return profile.has(ER_HLNS_IDLE_ROUTINE_DISPATCH_CAPABILITY)
    cfg = getattr(env, "cfg", None)
    return bool(
        getattr(cfg, "hrl_route_plan_idle_routine_dispatch_enabled", False)
    )


def er_hlns_balanced_all_tasks_active(env: Any) -> bool:
    """Return whether the candidate-only balanced-all-tasks pilot is active."""

    profile = getattr(env, "algorithm_profile", None)
    if isinstance(profile, AlgorithmProfile):
        return profile.has(ER_HLNS_BALANCED_ALL_TASKS_CAPABILITY)
    cfg = getattr(env, "cfg", None)
    return bool(
        getattr(cfg, "hrl_route_plan_balanced_all_tasks_enabled", False)
    )


def er_hlns_balanced_all_tasks_v2_active(env: Any) -> bool:
    """Return whether the candidate-only post-launch V2 pilot is active."""

    profile = getattr(env, "algorithm_profile", None)
    if isinstance(profile, AlgorithmProfile):
        return profile.has(ER_HLNS_BALANCED_ALL_TASKS_V2_CAPABILITY)
    cfg = getattr(env, "cfg", None)
    return bool(
        getattr(cfg, "hrl_route_plan_balanced_all_tasks_v2_enabled", False)
    )


def er_hlns_balanced_all_tasks_v3_active(env: Any) -> bool:
    """Return whether the candidate-only dual-plan selector pilot is active."""

    profile = getattr(env, "algorithm_profile", None)
    if isinstance(profile, AlgorithmProfile):
        return profile.has(ER_HLNS_BALANCED_ALL_TASKS_V3_CAPABILITY)
    cfg = getattr(env, "cfg", None)
    return bool(
        getattr(cfg, "hrl_route_plan_balanced_all_tasks_v3_enabled", False)
    )


def er_hlns_balanced_all_tasks_v5_active(env: Any) -> bool:
    """Return whether the candidate-only launch-first V5 pilot is active.

    V5 deliberately reuses the V3 capability and corridor implementation; the
    extra configuration flag is the only opt-in for its initial-route
    promotion.  This keeps the formal ER-HLNS profile unchanged.
    """

    profile = getattr(env, "algorithm_profile", None)
    if isinstance(profile, AlgorithmProfile) and not profile.has(
        ER_HLNS_BALANCED_ALL_TASKS_V3_CAPABILITY
    ):
        return False
    cfg = getattr(env, "cfg", None)
    return bool(
        getattr(
            cfg,
            "hrl_route_plan_balanced_all_tasks_v5_launch_first_enabled",
            False,
        )
    )


def er_hlns_low_seed_rescue_active(env: Any) -> bool:
    """Return whether the explicit low-seed candidate repair is active."""

    profile = getattr(env, "algorithm_profile", None)
    if isinstance(profile, AlgorithmProfile):
        return profile.has(ER_HLNS_LOW_SEED_RESCUE_CAPABILITY)
    return False
