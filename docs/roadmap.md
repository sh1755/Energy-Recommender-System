# Six-week research roadmap

This is a realistic pilot-study schedule for one researcher working regularly; publication and peer review are additional time. The supplied package accelerates implementation but does not replace real-data validation.

| Week | Work | Exit criterion |
|---|---|---|
| 1 | Download LCL, check licence, cohort and timezone assumptions; audit missingness, duplicate conflicts and tariff groups | Versioned dataset manifest, quality report, fixed cohort and temporal boundaries |
| 2 | Examine weekly profiles; test behavioural score definitions and missing-data sensitivity; define relevance without test tuning | Documented score rationale, confidence calibration and frozen preprocessing |
| 3 | Run baselines and ALS with expanded validation grids across several temporal folds; characterize cold-start fallback | Reproducible model tables with compute cost and variability |
| 4 | Replace illustrative task schedules with defensible appliance scenarios or participant constraints; align historical tariff/calendar | Auditable task-level counterfactual simulator and sensitivity bounds |
| 5 | Run ablations, paired household/block bootstrap, seasonal generalization and robustness tests; investigate rebound and unfair outcomes | Frozen primary results, effect sizes, uncertainty and negative findings |
| 6 | Draft methods/results/limitations; finalize figures and code release; independently reproduce from a fresh environment | Complete manuscript draft and reproducibility bundle |

## Future dataset extensions (not implemented)

**Ausgrid Solar Homes:** add import/export accounting, solar self-consumption and household PV characteristics. Distinguish gross generation, consumption and net imports. Future objective should use purchased energy rather than assuming all shifted demand is grid supplied. Verify current access and licence at the provider before reuse: https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data . This is a provider reference, not a verified automatic downloader.

**REFIT:** use appliance-level measurements to identify candidate task windows, durations and power traces. Aggregate watts over actual time intervals to kWh; do not treat instantaneous power as half-hour energy. Validate inferred cycles against labels and explicitly handle overlap. Source: https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements/ .

**ACN-Data:** use connection time, disconnection time and requested/delivered energy to represent EV availability and charging constraints. Add charge power limits and deadlines; evaluate unmet energy, not just low cost. Site-level charging records should not be mislabelled residential household behaviour. Source: https://ev.caltech.edu/dataset .
