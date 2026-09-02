from types import SimpleNamespace

from hetgat_hrl.agents.actor_critic import RuleBasedLowLevelPolicy
from hetgat_hrl.core.mdp_spec import AgentKind, EnvConfig
from hetgat_hrl.core.runtime_constants import DEPOT_DOCK_ID
from hetgat_hrl.envs.base_env import BaseHeteroDisasterEnv
from hetgat_hrl.hrl.rolling_planner import EventTriggeredRollingPlanner
from hetgat_hrl.training.runner import EpisodeRunner


def test_rolling_planner_smoke() -> None:
    env = BaseHeteroDisasterEnv(
        EnvConfig(
            num_trucks=2,
            num_uavs=2,
            num_nodes=20,
            num_edges=30,
            num_normal_tasks=2,
            num_emergency_tasks=2,
            max_steps=20,
            hrl_interval=3,
        )
    )
    low = RuleBasedLowLevelPolicy(seed=0)
    high = EventTriggeredRollingPlanner(decision_interval=3, seed=0)
    runner = EpisodeRunner(env=env, low_policy=low, high_planner=high)
    m = runner.run_episode(seed=0)
    assert m.steps > 0
    assert 0.0 <= m.task_completion_rate <= 1.0


def test_depot_dock_goal_is_not_classified_as_missing_goal() -> None:
    """The UAV depot/reload sentinel is a valid non-task recovery goal."""
    env = SimpleNamespace(
        state=SimpleNamespace(
            step_index=12,
            agents={
                "uav_0": SimpleNamespace(
                    kind=AgentKind.UAV,
                    crashed=False,
                    battery=0.2,
                )
            },
            tasks={},
        )
    )
    planner = EventTriggeredRollingPlanner(decision_interval=3, seed=0)
    planner.state.goals = {"uav_0": DEPOT_DOCK_ID}

    assert planner._goal_invalidated_refresh(env) is False
    assert planner._last_goal_invalid_record is None

    # Preserve the existing behavior for an actually missing task goal.
    planner.state.goals = {"uav_0": "missing-task"}
    assert planner._goal_invalidated_refresh(env) is True
    assert planner._last_goal_invalid_record["reason"] == "task_missing"

