"""Recompute principal episode summaries and paired E1 contrasts.

Run from the root of the public reproducibility package:
    python analysis/reproduce_public_summaries.py --output reproduced
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

from paired_bootstrap import percentile_interval


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data"
METRICS = {
    "overall": "overall_completion_rate",
    "routine": "routine_bulk_completion_rate",
    "time_critical": "time_critical_lightweight_completion_rate",
    "PWCT": "pwct_seconds",
    "runtime": "runtime_seconds",
}

E1_METHODS = {
    "PR": "ER-HLNS-parallel-rescue",
    "CMP": "C-ALNS-MP",
    "C": "C-ALNS",
    "bare": "ER-HLNS",
}
E1_COMPARISONS = {
    "PR-CMP": ("PR", "CMP"),
    "CMP-C": ("CMP", "C"),
    "PR-bare": ("PR", "bare"),
}


def group_summary(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_key, group in frame.groupby(keys, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        base = dict(zip(keys, group_key))
        for metric, column in METRICS.items():
            values = pd.to_numeric(group[column]).to_numpy(float)
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "n_seed_episodes": len(values),
                    "mean": values.mean(),
                    "sd": values.std(ddof=1) if len(values) > 1 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def e1_paired(frame: pd.DataFrame) -> pd.DataFrame:
    e1 = frame[(frame["family"] == "E1") & frame["condition"].isin(["island_on", "island_off"])]
    rows: list[dict[str, object]] = []
    for condition in ("island_on", "island_off"):
        subset = e1[e1["condition"] == condition]
        for metric, column in METRICS.items():
            pivot = subset.pivot(index="seed", columns="method", values=column)
            diff = (
                pd.to_numeric(pivot["ER-HLNS-parallel-rescue"])
                - pd.to_numeric(pivot["C-ALNS-MP"])
            ).to_numpy(float)
            low, high = percentile_interval(
                diff, f"E1|{condition}|PR-CMP|{metric}", namespace="bootstrap"
            )
            favourable = diff if metric not in {"PWCT", "runtime"} else -diff
            rows.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "n_paired_seeds": len(diff),
                    "paired_mean": diff.mean(),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "wins": int((favourable > 0).sum()),
                    "ties": int((favourable == 0).sum()),
                    "losses": int((favourable < 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def e1_all_paired(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute every reported E1 contrast with the canonical bootstrap."""
    e1 = frame[(frame["family"] == "E1") & frame["condition"].isin(["island_on", "island_off"])]
    rows: list[dict[str, object]] = []
    for condition in ("island_on", "island_off"):
        for comparison, (target_key, baseline_key) in E1_COMPARISONS.items():
            target = e1[(e1["condition"] == condition) & (e1["method"] == E1_METHODS[target_key])]
            baseline = e1[(e1["condition"] == condition) & (e1["method"] == E1_METHODS[baseline_key])]
            left = target.set_index("seed")
            right = baseline.set_index("seed")
            seeds = sorted(set(left.index) & set(right.index))
            for metric, field in METRICS.items():
                diffs = np.asarray(
                    [float(left.loc[seed, field]) - float(right.loc[seed, field]) for seed in seeds],
                    dtype=float,
                )
                low, high = percentile_interval(
                    diffs,
                    f"E1|{condition}|{comparison}|{metric}",
                    namespace="bootstrap",
                )
                higher = metric not in {"PWCT", "runtime"}
                wins = int(np.sum(diffs > 1e-12)) if higher else int(np.sum(diffs < -1e-12))
                losses = int(np.sum(diffs < -1e-12)) if higher else int(np.sum(diffs > 1e-12))
                rows.append(
                    {
                        "family": "E1",
                        "condition": condition,
                        "comparison": comparison,
                        "target": E1_METHODS[target_key],
                        "baseline": E1_METHODS[baseline_key],
                        "metric": metric,
                        "n_pairs": len(diffs),
                        "paired_mean": float(np.mean(diffs)),
                        "paired_sd": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
                        "bootstrap_replicates": 10_000,
                        "bootstrap_ci_low": low,
                        "bootstrap_ci_high": high,
                        "wins": wins,
                        "ties": len(diffs) - wins - losses,
                        "losses": losses,
                        "difference_direction": "target_minus_baseline",
                        "better_direction": "higher" if higher else "lower",
                        "sign_flip_p_exploratory": exact_sign_flip(diffs.tolist()),
                        "inference_label": "exploratory_only; not confirmatory",
                    }
                )
    return pd.DataFrame(rows)


EXTENSION_METRICS = {
    "overall_completion": "overall_completion_rate",
    "routine_completion": "routine_bulk_completion_rate",
    "tc_completion": "time_critical_lightweight_completion_rate",
    "tc_on_time_completion": "time_critical_on_time_completion_rate",
    "pwct_seconds": "pwct_seconds",
    "runtime_seconds": "runtime_seconds",
    "objective_evaluations": "alns_objective_evaluation_count",
}


def attach_pwct(frame: pd.DataFrame, task_path: Path, horizon: float = 28_800.0) -> pd.DataFrame:
    """Add execution-derived PWCT to a seed-metrics frame."""
    tasks = pd.read_csv(task_path)
    grouped = {key: block for key, block in tasks.groupby(["family", "condition", "method", "seed"], dropna=False)}
    values: list[float] = []
    for _, row in frame.iterrows():
        key = (row["family"], row["condition"], row["method"], int(row["seed"]))
        block = grouped.get(key, tasks.iloc[0:0])
        weight = pd.to_numeric(block["urgency_score"], errors="coerce")
        completed = block["completed"].astype(str).str.lower().isin(["1", "true"])
        completion = pd.to_numeric(block["completed_seconds"], errors="coerce").fillna(horizon).clip(upper=horizon)
        values.append(float((weight * completion.where(completed, horizon)).sum() / weight.sum()))
    result = frame.copy()
    result["pwct_seconds"] = values
    return result


def validate_e5_task_ledger(metrics: pd.DataFrame, task_path: Path) -> None:
    """Check the public E5 task ledger is one row per task and matches episodes."""
    tasks = pd.read_csv(task_path)
    assert len(tasks) == 200, f"expected 200 public E5 task rows, got {len(tasks)}"
    key_columns = ["family", "condition", "method", "seed", "task_id"]
    assert not tasks.duplicated(key_columns).any(), "duplicate E5 task key"
    counts = tasks.groupby("seed", dropna=False)["task_id"].nunique().to_dict()
    assert counts == {seed: 20 for seed in range(100, 110)}, f"unexpected E5 task counts: {counts}"
    expected_hash = metrics.set_index("seed")["algorithm_config_hash"].astype(str).to_dict()
    for seed, block in tasks.groupby("seed", dropna=False):
        hashes = set(block["algorithm_config_hash"].astype(str))
        assert hashes == {expected_hash[int(seed)]}, (
            f"E5 task/config mismatch for seed {seed}: {hashes} != {expected_hash[int(seed)]}"
        )


def exact_sign_flip(values: list[float]) -> float:
    observed = abs(float(np.mean(values)))
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        extreme += int(abs(float(np.mean([v * s for v, s in zip(values, signs)]))) >= observed - 1e-15)
    return extreme / (2 ** len(values))


def paired_extension(target: pd.DataFrame, baseline: pd.DataFrame, metric: str, label: str) -> dict[str, object] | None:
    field = EXTENSION_METRICS[metric]
    left = target.set_index("seed")[field].astype(float)
    right = baseline.set_index("seed")[field].astype(float)
    seeds = sorted(set(left.index) & set(right.index))
    diffs = [float(left[s] - right[s]) for s in seeds if math.isfinite(left[s]) and math.isfinite(right[s])]
    if not diffs:
        return None
    low, high = percentile_interval(np.asarray(diffs, dtype=float), f"{label}|{metric}")
    higher = metric not in {"pwct_seconds", "runtime_seconds", "objective_evaluations"}
    wins = sum((d > 0) if higher else (d < 0) for d in diffs)
    losses = sum((d < 0) if higher else (d > 0) for d in diffs)
    return {
        "contrast": label,
        "metric": metric,
        "n_paired_seeds": len(diffs),
        "paired_mean": float(np.mean(diffs)),
        "bootstrap_interval_95_low": low,
        "bootstrap_interval_95_high": high,
        "exact_sign_flip_p": exact_sign_flip(diffs),
        "wins": wins,
        "ties": len(diffs) - wins - losses,
        "losses": losses,
    }


def extension_group_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (family, condition, method), group in frame.groupby(["family", "condition", "method"], dropna=False):
        record: dict[str, object] = {"family": family, "condition": condition, "method": method, "n_seeds": len(group)}
        for metric, field in EXTENSION_METRICS.items():
            values = pd.to_numeric(group[field], errors="coerce").dropna()
            record[f"{metric}_mean"] = float(values.mean()) if len(values) else ""
            record[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) else ""
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["family", "condition", "method"])


def mechanism_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = pd.read_csv(frame)
    grouped: list[dict[str, object]] = []
    for (condition, method, seed), block in rows.groupby(["condition", "method", "seed"], dropna=False):
        churn: dict[str, int] = {}
        for text in block.get("owner_churn_task_ids", pd.Series(dtype=str)).fillna(""):
            for task_id in str(text).split("|"):
                if task_id:
                    churn[task_id] = churn.get(task_id, 0) + 1
        record: dict[str, object] = {"family": "E4", "condition": condition, "method": method, "seed": int(seed),
            "repeated_reassigned_task_count": sum(v >= 2 for v in churn.values()),
            "reassignment_events_beyond_first": sum(max(v - 1, 0) for v in churn.values())}
        for field in ["route_edit_distance", "owner_churn", "execution_rebind", "invalidated_commitment_count",
                      "recovery_plan_change_count", "recovery_plan_invalidation_count", "claimed_protected_changed",
                      "airborne_protected_changed", "repair_wall_ms"]:
            record[field] = float(pd.to_numeric(block[field], errors="coerce").fillna(0).sum())
        grouped.append(record)
    return pd.DataFrame(grouped)


def mechanism_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["route_edit_distance", "owner_churn", "execution_rebind", "invalidated_commitment_count",
               "recovery_plan_change_count", "recovery_plan_invalidation_count", "claimed_protected_changed",
               "airborne_protected_changed", "repeated_reassigned_task_count", "reassignment_events_beyond_first",
               "repair_wall_ms"]
    left = frame[frame["condition"] == "suffix_local"].set_index("seed")
    right = frame[frame["condition"] == "global_pending_reauction"].set_index("seed")
    output: list[dict[str, object]] = []
    for metric in metrics:
        seeds = sorted(set(left.index) & set(right.index))
        diffs = [float(left.loc[s, metric] - right.loc[s, metric]) for s in seeds]
        if not diffs:
            continue
        low, high = percentile_interval(np.asarray(diffs), f"E4|mechanism|{metric}")
        output.append({"contrast": "E4_suffix_local_minus_global_reauction", "metric": metric,
                       "n_paired_seeds": len(diffs), "paired_mean": float(np.mean(diffs)),
                       "bootstrap_interval_95_low": low, "bootstrap_interval_95_high": high,
                       "exact_sign_flip_p": exact_sign_flip(diffs),
                       "wins": sum(d < 0 for d in diffs), "ties": sum(d == 0 for d in diffs),
                       "losses": sum(d > 0 for d in diffs), "better_direction": "lower"})
    return pd.DataFrame(output)


SAFETY_FIELDS = [
    "terminal_battery_rescue",
    "runtime_crash_count",
    "uav_drop_count",
    "crash_count",
    "physical_v2_uav_drop_count",
    "physical_v2_forced_landing_count",
    "physical_v2_uav_drop_runtime_count",
    "UAV_DROP",
    "FORCED_LANDING",
    "MISSION_ABORT",
    "ENERGY_EXHAUSTION",
    "RECOVERY_FAILURE",
    "uav_airborne_safety_abort_count",
    "hard_constraint_violation_count",
    "hard_failure_component_sum_not_unique",
    "hard_failure_any",
]


def _ensure_safety_fields(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "terminal_battery_rescue" not in result:
        raw_name = "uav_terminal_battery_rescue_count_total"
        result["terminal_battery_rescue"] = (
            pd.to_numeric(result[raw_name], errors="coerce").fillna(0)
            if raw_name in result
            else 0.0
        )
    for field in SAFETY_FIELDS[1:14]:
        if field not in result:
            result[field] = 0.0
        result[field] = pd.to_numeric(result[field], errors="coerce").fillna(0.0)
    if "hard_failure_component_sum_not_unique" not in result:
        result["hard_failure_component_sum_not_unique"] = result[SAFETY_FIELDS[1:14]].sum(axis=1)
    else:
        result["hard_failure_component_sum_not_unique"] = pd.to_numeric(
            result["hard_failure_component_sum_not_unique"], errors="coerce"
        ).fillna(0.0)
    result["hard_failure_any"] = (
        result["hard_failure_component_sum_not_unique"] > 0
    ).astype(int)
    return result


def safety_audit(paths: list[Path], output: Path) -> None:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = _ensure_safety_fields(pd.read_csv(path))
        frame.insert(0, "source_file", path.name)
        frames.append(frame)
    safety = pd.concat(frames, ignore_index=True, sort=False)
    assert len(safety) == 780, f"expected 780 safety-audited executions, got {len(safety)}"
    hard = int(pd.to_numeric(safety["hard_failure_any"], errors="coerce").fillna(0).sum())
    assert hard == 0, f"expected zero hard outcomes across 780 executions, got {hard}"
    original = safety[safety["source_file"] == "SourceData_safety_guard_summary.csv"]
    rescue = int(pd.to_numeric(original["terminal_battery_rescue"], errors="coerce").fillna(0).sum())
    assert len(original) == 400, f"expected 400 original-family safety rows, got {len(original)}"
    assert rescue == 612, f"expected 612 original-family terminal-battery rescues, got {rescue}"
    # The original robustness safety summary retains two blockage settings for
    # a family/scenario/method/seed combination but does not retain the setting
    # label. Preserve a deterministic within-key ordinal so every emitted row
    # remains addressable without inferring a nonexistent condition value.
    row_key = ["family", "scenario", "method", "seed"]
    safety["episode_occurrence_index"] = (
        safety.groupby(row_key, sort=False, dropna=False).cumcount().add(1).astype(int)
    )
    columns = [
        "source_file", "family", "condition", "scenario", "method", "seed",
        "episode_occurrence_index", *SAFETY_FIELDS
    ]
    for column in columns:
        if column not in safety:
            safety[column] = ""
    safety[columns].to_csv(output / "safety_guard_summary.csv", index=False)


def public_task_scope(extended: pd.DataFrame, e4: pd.DataFrame, e5: pd.DataFrame, e6e7: pd.DataFrame, output: Path) -> None:
    """Write a machine-readable statement of public versus internal task scope."""
    assert len(extended) == 220, f"expected 220 public E1-E3 episode rows, got {len(extended)}"
    rows = [
        {
            "dataset": "E1-E3",
            "episode_rows_public": len(extended),
            "task_rows_public": 0,
            "task_rows_internal_audit": 4400,
            "task_ledger_public": False,
            "note": "Episode-level source is public; nested task ledger was audited internally and is not redistributed.",
        },
        {
            "dataset": "E4",
            "episode_rows_public": len(e4),
            "task_rows_public": 400,
            "task_rows_internal_audit": 400,
            "task_ledger_public": True,
            "note": "One public row per task in the direct repair-scope control.",
        },
        {
            "dataset": "E5",
            "episode_rows_public": len(e5),
            "task_rows_public": 200,
            "task_rows_internal_audit": 200,
            "task_ledger_public": True,
            "note": "One public row per task after seed-103 configuration de-duplication.",
        },
        {
            "dataset": "E6-E7",
            "episode_rows_public": len(e6e7),
            "task_rows_public": 1600,
            "task_rows_internal_audit": 1600,
            "task_ledger_public": True,
            "note": "Conditional truck-eligibility and lifeline-sensitivity task rows.",
        },
    ]
    pd.DataFrame(rows).to_csv(output / "public_task_ledger_scope.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reproduced")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    main_data = pd.read_csv(SOURCE / "SourceData_Figure4_main_five_scenario.csv")
    extended = pd.read_csv(SOURCE / "SourceData_extended_episode_metrics.csv")
    group_summary(main_data, ["scenario", "method"]).to_csv(
        output / "main_five_scenario_group_summary.csv", index=False
    )
    group_summary(extended, ["family", "condition", "scenario", "method"]).to_csv(
        output / "extended_group_summary.csv", index=False
    )
    e1_paired(extended).to_csv(output / "E1_PR_minus_C_ALNS_MP.csv", index=False)
    e1_all_paired(extended).to_csv(output / "E1_all_paired_contrasts.csv", index=False)
    e4 = attach_pwct(pd.read_csv(SOURCE / "SourceData_E4_seed_metrics.csv"), SOURCE / "SourceData_E4_task_outcomes.csv")
    e5_metrics = pd.read_csv(SOURCE / "SourceData_E5_seed_metrics_budget_capped.csv")
    validate_e5_task_ledger(e5_metrics, SOURCE / "SourceData_E5_task_outcomes.csv")
    e5 = attach_pwct(e5_metrics, SOURCE / "SourceData_E5_task_outcomes.csv")
    e6e7 = attach_pwct(pd.read_csv(SOURCE / "SourceData_E6_E7_seed_metrics.csv"), SOURCE / "SourceData_E6_E7_task_outcomes.csv")
    extensions = pd.concat([e4, e5, e6e7], ignore_index=True)
    extension_group_summary(extensions).to_csv(output / "extension_group_summary.csv", index=False)

    e4_group = extension_group_summary(e4)
    e4_group.to_csv(output / "E4_repair_scope_summary.csv", index=False)
    extension_group_summary(e5).to_csv(output / "E5_equal_budget_summary.csv", index=False)
    extension_group_summary(e6e7[e6e7["family"] == "E6"]).to_csv(output / "E6_conditional_eligibility_summary.csv", index=False)
    extension_group_summary(e6e7[e6e7["family"] == "E7"]).to_csv(output / "E7_lifeline_sensitivity_summary.csv", index=False)

    paired_rows: list[dict[str, object]] = []
    pairs = [
        (e4[(e4.family == "E4") & (e4.condition == "suffix_local")], e4[(e4.family == "E4") & (e4.condition == "global_pending_reauction")], "E4_suffix_local_minus_global_reauction"),
        (e4[(e4.family == "E4") & (e4.condition == "suffix_local")], e5, "E5_ER_HLNS_minus_equal_budget_C_ALNS"),
        (e6e7[(e6e7.family == "E6") & (e6e7.method == "ER-HLNS-parallel-rescue")], e6e7[(e6e7.family == "E6") & (e6e7.method == "C-ALNS-MP")], "E6_ER_HLNS_minus_C_ALNS_conditional_truck_TC"),
        (e6e7[(e6e7.family == "E6") & (e6e7.method == "ER-HLNS-parallel-rescue")], e4[(e4.family == "E4") & (e4.condition == "suffix_local")], "E6_ER_HLNS_conditional_minus_strict_TC_eligibility"),
    ]
    for target, baseline, label in pairs:
        for metric in EXTENSION_METRICS:
            row = paired_extension(target, baseline, metric, label)
            if row:
                paired_rows.append(row)
    for method in ("C-ALNS-MP", "ER-HLNS-parallel-rescue"):
        nominal = e6e7[
            (e6e7.family == "E7")
            & (e6e7.condition == "beta_c_0.22")
            & (e6e7.method == method)
        ]
        for beta in ("0.16", "0.28"):
            changed = e6e7[
                (e6e7.family == "E7")
                & (e6e7.condition == f"beta_c_{beta}")
                & (e6e7.method == method)
            ]
            for metric in EXTENSION_METRICS:
                row = paired_extension(changed, nominal, metric, f"E7_{method}_beta_{beta}_minus_0.22")
                if row:
                    paired_rows.append(row)
    for beta in ("0.16", "0.22", "0.28"):
        target = e6e7[(e6e7.family == "E7") & (e6e7.condition == f"beta_c_{beta}") & (e6e7.method == "ER-HLNS-parallel-rescue")]
        baseline = e6e7[(e6e7.family == "E7") & (e6e7.condition == f"beta_c_{beta}") & (e6e7.method == "C-ALNS-MP")]
        for metric in EXTENSION_METRICS:
            row = paired_extension(target, baseline, metric, f"E7_ER_HLNS_minus_C_ALNS_beta_{beta}")
            if row:
                paired_rows.append(row)
    pd.DataFrame(paired_rows).to_csv(output / "extension_paired_contrasts.csv", index=False)

    mechanism_path = SOURCE / "SourceData_E4_execution_mechanism_rows.csv"
    if not mechanism_path.exists():
        mechanism_path = mechanism_path.with_suffix(".csv.gz")
    mech = mechanism_summary(mechanism_path)
    mechanism_contrasts(mech).to_csv(output / "E4_mechanism_contrasts.csv", index=False)
    mech.to_csv(output / "E4_mechanism_seed_summary.csv", index=False)
    public_task_scope(extended, e4, e5, e6e7, output)
    safety_audit(
        [
            SOURCE / "SourceData_safety_guard_summary.csv",
            SOURCE / "SourceData_extended_episode_metrics.csv",
            SOURCE / "SourceData_strict_TO_seed_metrics.csv",
            SOURCE / "SourceData_E4_seed_metrics.csv",
            SOURCE / "SourceData_E5_seed_metrics_budget_capped.csv",
            SOURCE / "SourceData_E6_E7_seed_metrics.csv",
        ],
        output,
    )
    print(f"Wrote summaries to {output}")


if __name__ == "__main__":
    main()
