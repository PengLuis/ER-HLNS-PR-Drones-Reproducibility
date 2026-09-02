SOURCE DATA FOR THE DRONES MANUSCRIPT

Independent unit: one method-condition-seed episode.
Paired seeds: 100-109 (n=10 paired seed blocks per contrast).
Nested task rows are used only to construct episode outcomes and are not independent replicates.

Formal scenario and seed protocol: seeds 100-109 were applied unchanged to all paired methods and conditions. Each seed generates a distinct M- or L-scale synthetic road-network instance; M-MB and L-LB share the seed-specific topology and task realization within each paired control/disruption comparison. RB retains one fixed Dujiangyan-derived base topology and fixed task manifest; its seeds vary the dynamic/persistent disruption realization and other code-defined episode-level exogenous draws, not the base map.

The CSV files contain the values underlying Figures 4-6, Tables 4-8, strict-eligibility truck-only results, Supplementary Tables S2-S12, the E4-E7 controls, and the depot-sentinel regression audit. No planned unique episode or task outcome was excluded. The E5 task ledger is de-duplicated to one row per task: the 20 duplicate seed-103 rows carrying an orphaned configuration hash were removed as public-file sanitation because the E5 episode ledger records the retained configuration hash; the retained one-row-per-task outcomes are unchanged. Marginal 95% intervals use the canonical deterministic 10,000-resample seed-block bootstrap implementation shared by the analysis generator and public reproduction script. The canonical seed is the full 64-bit integer represented by the first eight SHA-256 digest bytes; no historical 32-bit truncation is used. The E1-E7 analyses are exploratory and are not simultaneous or multiplicity-adjusted unless a table explicitly states otherwise.

SourceData_Figure4_main_five_scenario.csv contains one row for every main method-scenario-seed episode. Its completed_tasks field is the sum of the recorded routine-bulk and time-critical-lightweight completed-task counts, and weighted_service_score is the episode-level quantity summarized in Table 4. For task i, let u_i be urgency clipped to [0,1] and v_i be the fulfilled-demand fraction for a routine task, the remaining-lifeline fraction at service for a delivered TC task, or zero for an unfinished TC task; the reported score is Σ_i u_i v_i / Σ_i max(u_i, 10^-6), with each v_i clipped to [0,1]. Figure 4-6 filenames follow the current manuscript numbering.

Method display names:
- ER-HLNS-PR: ER-HLNS with the parallel-rescue overlay.
- C-ALNS-MP: canonical interval-only ALNS with the leading mission-priority rank matched to the proposed controller; its 14-component route representation is not identical.
- E5 budget control: C-ALNS-MP is stopped at a seed-specific objective-evaluation ceiling equal to the suffix-local ER-HLNS-PR reference; target and realized counts are both retained, including shortfalls.

E4-E7 files:
- SourceData_E4_seed_metrics.csv and SourceData_E4_task_outcomes.csv: direct suffix-local/global repair control.
- SourceData_E4_execution_mechanism_rows.csv.gz: gzip-compressed event-level repair, owner-churn, rebind, and commitment ledger. Pandas reads this file directly.
- SourceData_E5_seed_metrics_budget_capped.csv and SourceData_E5_task_outcomes.csv: budget-capped C-ALNS-MP control. The task ledger contains 200 rows (10 seeds x 20 unique tasks); for seed 103 the retained hash is `d5bc065260648006cbe8e68287278c564739732db79cad0987488825d5ac0be4`, matching the episode-level ledger. The removed duplicate hash was `4cf8c1e71a12e510147cf39beb12c283442385f80907a799b7cfdb1813abca36`; its rows did not change the recomputed E5 PWCT.
- SourceData_E6_E7_seed_metrics.csv and SourceData_E6_E7_task_outcomes.csv: conditional-truck and lifeline-sensitivity episodes.
- SourceData_E4_E5_E6_E7_group_summary.csv and SourceData_E4_E5_E6_E7_paired_contrasts.csv: deterministic summaries and paired contrasts.
- SourceData_E6_conditional_truck_tc_seed_summary.csv: task-level TC completing-agent counts.

Safety audit coverage:
- SourceData_extended_episode_metrics.csv (220 E1-E3 rows) and SourceData_strict_TO_seed_metrics.csv (50 TO rows) include the per-execution hard-outcome fields.
- The E4, E5, and E6-E7 seed ledgers retain the same hard-outcome components.
- Together with the 400-row original-family safety summary, these ledgers cover all 780 analyzed executions.
- `SourceData_safety_guard_summary.csv` uses `(family, scenario, method, seed, episode_occurrence_index)` as its row key. The two retained robustness settings share the first four identifiers; the safety-only ledger does not retain their blockage-asymptote label, so `episode_occurrence_index` distinguishes the two retained records without inferring a condition value.

Task-ledger scope:
- The internal E1-E3 audit used 4,400 nested task rows (220 episodes x 20 tasks), but those task-level rows are not included in this public package; the public E1-E3 file is episode-level only.
- Public E4, E5, and E6-E7 task ledgers contain 400, 200, and 1,600 rows, respectively, after E5 de-duplication. Thus, 2,200 extension task rows are public; the manuscript's 6,600-row extension total refers to the complete internal audit and should not be read as a claim that all 6,600 raw rows are redistributed here.

The public reproducibility archive's `analysis/reproduce_public_summaries.py`
recomputes the E1 paired contrasts (all three reported comparison families) and
the E4-E7 group and paired summaries from the seed/task ledgers and
writes separate E4 repair-scope, E4 mechanism, E5 budget, E6 eligibility, and
E7 lifeline outputs. It also writes a 780-row `safety_guard_summary.csv`,
asserting zero hard outcomes across all execution families and separately
checking that the original 400-row family summary sums to 612 terminal-battery
rescue interventions.

Service-accounting boundary: the common execution layer permits a loaded UAV to
begin a time-critical service while co-located with its carrier. Such a
mounted-service event is recorded as UAV-attributed service but is not an
airborne launch. The released aggregate ledgers do not retain airborne/docked
state at service start; therefore UAV-delivery and complete launch-gate counters
must not be treated as interchangeable or used to infer a transition-level
airborne-service denominator.

Bootstrap metadata mapping: SourceData_Figure5_E1_paired_contrasts.csv, SourceData_Figure6_E2_physical_sensitivity.csv, SourceData_TableS3_ablation_contrasts.csv, and SourceData_FigureS1_blockage_contrasts.csv use the same canonical deterministic 10,000-resample paired seed-block percentile-bootstrap implementation described above. The last two files retain contrast values rather than a repeated `bootstrap_replicates` column; this README is the file-level metadata record.

Mechanism field definitions:
- planner_refresh_total_including_initialization includes one initial plan row per seed.
- event_driven_refresh_admitted excludes initialization and is the numerator for admission rates.
- prior_ledger_visible_contract_signatures_preserved_fraction is an audit signature measure, not a protected-owner invariant test.
- claimed_or_airborne_tasks_in_changed_suffix and airborne_tasks_in_changed_suffix are broad overlap diagnostics, not violation counts.

Regression-audit note:
- The valid depot-recovery sentinel is explicitly excluded from the missing-task refresh pathway.
- The regression audit found zero depot-sentinel task-missing classifications.
- Planner-side high-priority refresh triggers are not hard safety outcomes.
- Seed ledgers expose the experiment-runner digest under the `runner_sha256` field.

PWCT is priority-weighted censored completion time; unfinished tasks receive the scenario horizon.
