"""Public, provenance-neutral runner for the reported five-scenario experiments.

Place this file in ``scripts/`` beside ``experiment_matrix_methods.py`` and
``run_experiment_matrix.py``.  The sanitized scenario configuration is read
from ``../configs/paper_five_scenarios_public.json``.  The runner does not
depend on internal freeze manifests or local revision identifiers.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
from dataclasses import replace
from pathlib import Path
import sys
import time
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT))

from tools.experiment_matrix_methods import _build_algorithm_package  # noqa: E402
from hetgat_hrl.agents.actor_critic import RuleBasedLowLevelPolicy  # noqa: E402
from hetgat_hrl.core.mdp_spec import EnvConfig  # noqa: E402
from hetgat_hrl.envs.base_env import BaseHeteroDisasterEnv  # noqa: E402
from tools.run_experiment_matrix import _run_episode  # noqa: E402


CONFIG = PACKAGE_ROOT / "configs" / "paper_five_scenarios_public.json"
SCENARIOS = ("M", "MB", "L", "LB", "RB")
METHODS = {
    "PG": "priority_greedy",
    "Greedy": "greedy_rule",
    "C-ALNS": "rolling_horizon_alns",
    "DR-ALNS": "dynamic_replanning_alns",
    "ER-HLNS": "er_hlns",
    "ER-HLNS-PR": "er_hlns",
}
PARALLEL_OVERLAY = {
    "erc_ablate_low_value_refresh": False,
    "erc_ablate_normal_protection": False,
    "hrl_route_plan_mixed_coverage_enabled": True,
    "hrl_route_plan_mixed_coverage_emergency_reserve_steps": 30,
    "hrl_route_plan_residual_normal_bipartite_matching_enabled": True,
    "hrl_route_plan_routine_dynamic_reassignment_enabled": True,
    "hrl_route_plan_routine_dynamic_reassignment_radius_m": 800.0,
    "hrl_route_plan_routine_dynamic_reassignment_min_eta_gain_steps": 3.0,
    "hrl_route_plan_routine_dynamic_reassignment_min_eta_gain_ratio": 0.20,
    "hrl_route_plan_routine_dynamic_reassignment_max_transfers": 1,
    "hrl_route_plan_stalled_queue_rescue_enabled": True,
    "hrl_route_plan_stalled_queue_anchor_rescue_enabled": True,
    "hrl_route_plan_stalled_queue_rescue_normal_service_gate_enabled": True,
    "hrl_route_plan_stalled_queue_max_active_rescues": 3,
    "hrl_route_plan_queue_urgent_rescue_horizon_steps": 0,
    "hrl_route_plan_stalled_queue_rescue_steps": 18,
    "hrl_route_plan_stalled_queue_anchor_timeout_steps": 18,
    "hrl_route_plan_queue_rescue_launch_timeout_steps": 4,
}


def _load_config() -> dict[str, Any]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _scenario_values(token: str) -> tuple[str, dict[str, Any]]:
    frozen = _load_config()
    aliases = frozen.get("aliases", {})
    if token in frozen["scenarios"]:
        key = token
    else:
        inverse_aliases = {str(value): str(key) for key, value in aliases.items()}
        key = inverse_aliases.get(token, token)
    if key not in frozen["scenarios"]:
        raise KeyError(f"unknown scenario: {token}")
    return str(key), dict(frozen["scenarios"][key])


def _package(method: str, cfg: EnvConfig, seed: int, *, parallel: bool = False):
    package = _build_algorithm_package(method, cfg, int(seed))
    if parallel:
        package = replace(package, runtime_cfg=replace(package.runtime_cfg, **PARALLEL_OVERLAY))
    return package


def _strict_to_cfg(values: dict[str, Any]) -> EnvConfig:
    values = dict(values)
    values.update(
        {
            "num_uavs": 0,
            "n_uavs": 0,
            "truck_can_serve_emergency_tasks": False,
            "truck_conditional_emergency_service_enabled": False,
        }
    )
    return EnvConfig(**values)


def _scalar_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        str(k): v
        for k, v in row.items()
        if not str(k).startswith("_") and (isinstance(v, (str, int, float, bool)) or v is None)
    }


def _run_job(job: tuple[str, int, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scenario, seed, label = job
    scenario_key, values = _scenario_values(scenario)
    values["seed"] = int(seed)
    if label == "TO":
        cfg = _strict_to_cfg(values)
        package = _package("er_hlns", cfg, seed)
    else:
        cfg = EnvConfig(**values)
        package = _package(METHODS[label], cfg, seed, parallel=label == "ER-HLNS-PR")
    env = BaseHeteroDisasterEnv(package.runtime_cfg, algorithm_profile=package.profile)
    package.bind_environment(env)
    env.current_episode_index = 0
    started = time.perf_counter()
    result = _run_episode(
        env,
        package.planner,
        RuleBasedLowLevelPolicy(seed=int(seed)),
        eval_seed=int(seed),
        lightweight_metrics_only=False,
        audit_flags={"enable_event_ledger_detail": False, "enable_task_outcome_export": True},
    )
    row = _scalar_row(result)
    row.update(
        {
            "scenario": scenario_key,
            "method": label,
            "seed": int(seed),
            "runtime_seconds": float(time.perf_counter() - started),
        }
    )
    tasks: list[dict[str, Any]] = []
    for task in result.get("_task_outcome_rows", []):
        if isinstance(task, dict):
            item = dict(task)
            item.update({"scenario": scenario_key, "method": label, "seed": int(seed)})
            tasks.append(item)
    return row, tasks


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _parse_seeds(spec: str) -> list[int]:
    seeds: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            seeds.extend(range(int(left), int(right) + 1))
        else:
            seeds.append(int(token))
    return list(dict.fromkeys(seeds))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--seeds", default="100-109")
    parser.add_argument("--include-to", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")
    if args.include_to:
        methods.append("TO")
    jobs = [
        (scenario, seed, method)
        for seed in _parse_seeds(args.seeds)
        for scenario in [x.strip() for x in args.scenarios.split(",") if x.strip()]
        for method in methods
    ]
    rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    with mp.get_context("spawn").Pool(processes=max(1, args.workers)) as pool:
        for row, tasks in pool.imap_unordered(_run_job, jobs):
            rows.append(row)
            task_rows.extend(tasks)
            print(
                f"[{len(rows):03d}/{len(jobs):03d}] {row['scenario']} {row['method']} "
                f"seed={row['seed']} completion={float(row.get('overall_completion_rate', 0.0)):.3f}",
                flush=True,
            )
    rows.sort(key=lambda r: (int(r["seed"]), str(r["scenario"]), str(r["method"])))
    output = Path(args.output).resolve()
    _write_csv(output / "seed_metrics.csv", rows)
    _write_csv(output / "task_outcomes.csv", task_rows)


if __name__ == "__main__":
    mp.freeze_support()
    main()
