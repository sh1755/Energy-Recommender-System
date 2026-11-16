# Execution report — 14 November 2026

## Scope and provenance
Executed on Linux with Python 3.12.14. Results in this package use **synthetic demonstration data only**. No real LCL dataset was downloaded or evaluated. A constructed fixture with original LCL column names was ingested successfully; that validates the interface, not every irregularity in the original archive.

## Executed successfully
- Full pipeline: 36 households, 145,152 meter rows, seeds 11/22/33; eight model variants per seed, with actual Surprise UserKNN, ItemKNN, SVD and NMF implementations.
- 23 pytest cases: duplicate conflicts, missing values, temporal split order, frozen transformations, household-scale invariance, implicit learning and cold fallback, missing-context eligibility, score decomposition, energy conservation, model serialization, DST-aware timestamps, household overrides, absent-interval audit, invalid configuration/cohorts, fractional constraints, ID preservation, missing/subsecond timestamps, invalid task energy, empty evaluation diagnostics and configurable implicit target.
- Five notebooks / 19 code cells executed with in-process IPython; real outputs saved in each notebook. A socket-based Jupyter kernel could not start in this environment, so the supplied executor deliberately avoids kernel sockets. Interactive VS Code kernel operation was not tested here.
- Original-LCL-schema constructed fixture: 6 households, 6,912 rows, standalone preprocessing CLI, split files and affinity output.
- Independent saved-output checks: model row counts, exact score sums, task tariff savings, energy conservation across strategies, every availability/shift constraint and direct recomputation of an aggregate daily peak from raw held-out profiles.
- Six editable DOT architecture/process diagrams and seven result-based plots generated in PNG, SVG and PDF (39 figure exports total). Representative exported figures were visually inspected.

## Demonstration findings (not real-world evidence)
Mean NDCG@5 across seeds: contextual implicit ALS approximately 0.9835 and pooled implicit ALS 0.9721. UserKNN, ItemKNN and NMF reach 1.0 on these easy stylized profiles. These values are not LCL benchmarks and do not establish superiority, novelty or realistic recommendation acceptance.

Under the default assumed added 1 kWh task, hybrid and cost-only both save £0.16 per eligible augmented household-day versus the original slot. Mean household-day percentage saving is approximately 5.21% of the entire augmented bill. This percentage differs from “task percentage saving” and from a ratio-of-population-totals statistic. Hybrid does **not** outperform cost-only on price in this demonstration. It has a lower mean inconvenience proxy (about 0.783 versus 1.0 under the defined window). All exported constraints are satisfied.

Simulation includes 540 of 612 test household-days. Partial boundary and DST-length local days are excluded. Five cold households are held out. Ranking eligibility has 62 warm household/daytype contexts for each model and seed.

## Interpretation and outstanding work
The task is an exogenous addition to baseline meter load, not a detected appliance or all household demand. The tariff is simulated. Household bootstrap intervals are conditional on the synthetic scenario. Real LCL cohort execution, historical tariff alignment, seasonal evaluation, validated flexible appliances, user acceptance, online demand response and measured savings remain outstanding. Ausgrid, REFIT and ACN-Data are future directions only.

## Evidence files
`results/logs/pytest.txt`, `pipeline.txt`, `notebooks.txt`, `lcl_schema_smoke.txt`, `output_verification.txt`, `dashboard.txt`, `dependencies.txt` and `environment-freeze.txt`. Configuration and exact split timestamps are in `results/metadata.json`. The final dashboard and dependency smoke-test status is recorded below when packaging.

## Final packaging checks
The pinned Streamlit 1.41.1 / PyArrow 18.1.0 / Altair 5.5.0 combination was installed and verified. Streamlit AppTest rendered the dashboard and changed the household selector without exceptions. This is a headless application test; an interactive browser session was not tested. `pip check` reported no broken requirements. The final unit test rerun passed all 23 cases. The local package was built and installed in editable mode successfully.
