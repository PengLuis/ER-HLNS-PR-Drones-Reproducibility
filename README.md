# ER-HLNS-PR Drones reproducibility repository

This repository accompanies **“Execution-Consistent Event-Responsive Truck–UAV Coordination under Simulated Progressive Road Disruption.”** It is a curated research release containing the executable ER-HLNS-PR evaluation path, public experiment configurations, the fixed Dujiangyan-derived RB assets, processed source data, regression tests, and scripts for reconstructing the reported summaries.

## What is included

- `src/hetgat_hrl/`: planner, execution environment, and imported dependencies used by the public runner.
- `configs/`: frozen public scenario and algorithm settings.
- `scripts/run_public_experiment_suite.py`: runner for M, MB, L, LB, and RB.
- `data/real_maps_final/R_DJ_C/final/`: fixed RB road graph, POI candidates, and task manifest.
- `source_data/`: the values underlying the manuscript figures, tables, and robustness analyses.
- `analysis/`: deterministic paired-bootstrap and summary reconstruction code.
- `tests/`: focused execution-path regression tests.

This is a paper-focused reproducibility release, not the complete development repository. Compatibility modules retained under `src/` are inactive unless imported by the public runner.

## Environment

The reported runs used Windows, Python 3.9.23, NumPy 2.0.2, pandas 2.3.2, SciPy 1.13.1, NetworkX 3.2.1, and PyTorch 2.4.1+cu121. CUDA is not required for the rule-based evaluation path.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

On Linux or macOS, activate with `source .venv/bin/activate`.

## Verify the release

```bash
python tools/validate_rb_assets.py
python -m pytest tests/test_rolling_planner_smoke.py -q
python analysis/reproduce_public_summaries.py --output reproduced
```

The E4 event-level ledger is stored as `SourceData_E4_execution_mechanism_rows.csv.gz` to stay below GitHub's per-file size limit. Pandas reads it directly; no manual extraction is required.

## Re-run experiments

Synthetic M, MB, L, and LB scenarios:

```bash
python scripts/run_public_experiment_suite.py --scenarios M,MB,L,LB --seeds 100-109 --workers 6 --output rerun
```

Fixed Dujiangyan-derived RB scenario:

```bash
python scripts/run_public_experiment_suite.py --scenarios RB --seeds 100-109 --workers 6 --output rerun_rb
```

Use `--methods` to select a subset of methods and `--include-to` to include the strict truck-only modal-feasibility condition.

## Scenario and seed protocol

Seeds 100–109 are applied unchanged to paired methods and conditions. Each seed generates a distinct synthetic road-network instance at the M or L scale. Within a seed, M–MB and L–LB share the same topology, task realization, initial configuration, and common-random-number schedule.

RB uses one fixed Dujiangyan-derived base graph and one fixed 20-task manifest. Its seeds vary the dynamic disruption realization and other episode-level exogenous draws; they do not change the base topology or task locations. The checked-in RB assets contain 462 nodes, 709 undirected operational edges, 8 routine-bulk tasks, and 12 time-critical tasks.

## Statistical reproduction

The independent unit is one method–condition–seed episode. Tasks are nested and are not treated as independent replicates. Paired contrasts use seeds 100–109 (`n = 10`) and a deterministic 10,000-resample seed-block percentile bootstrap. The E1–E7 analyses are exploratory and intervals are marginal rather than simultaneous unless a table states otherwise.

`analysis/reproduce_public_summaries.py` reconstructs the E1 contrasts, E4–E7 group summaries and paired contrasts, task-ledger scope, and the 780-execution safety summary from the released ledgers.

## RB provenance and integrity

`RB_ASSET_MANIFEST.sha256` records the exact release hashes. `tools/validate_rb_assets.py` checks the hashes, graph dimensions, and task composition before an RB run.

The RB graph and POI metadata were derived from OpenStreetMap data and must retain OpenStreetMap attribution. See `DATA_AND_CODE_NOTICE.md` for provenance and rights information. The preparation script is included for transparency; recreating upstream OSM extracts requires the optional `osmnx`, `matplotlib`, and `PyYAML` dependencies.

## Interpretation notes

- `C-ALNS-MP` matches the leading mission-priority rank; it does not share the full ER-HLNS-PR route representation.
- A time-critical task can be served when the carrier reaches its location; the recorded serving agent follows the simulator's service-accounting convention.
- PWCT is priority-weighted censored completion time; unfinished tasks receive the scenario horizon.
- The source-data filenames follow the current manuscript numbering.

## Citation

If you use this repository, cite the associated manuscript. The final journal citation and DOI will be added after publication.
