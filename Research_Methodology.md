# Research methodology

An Explainable Hybrid Collaborative Filtering Framework for Personalised Residential Demand Response and Energy Cost Optimisation

Bundled outcomes are synthetic software-demonstration results, not measured Low Carbon London results.


---

# Mathematical formulation

## 1. Observation, context and scope
Let x(u,d,s) be observed electricity energy in kWh for household u, date d and half-hour slot s in {0,...,47}. It is energy, not power. Average power over that half hour is 2x kW. Let c(d) be 0 for weekday and 1 for weekend. Item i = 48c + s gives 96 repeatable scheduling contexts. Missing observations are excluded rather than equated to zero. This is repeat-consumption recommendation: previously observed items remain valid future candidates.

The derived quantity is **historical activity compatibility**, not revealed desire, controllability or an explicit household rating. A refrigerator, heating, occupancy, income or household size can drive aggregate readings. Low consumption may indicate absence or efficiency; high consumption may indicate an inflexible obligation. The affinity term therefore never determines flexibility constraints.

## 2. Bounded behavioural affinity
For each household, fit an activity threshold q35(u) and magnitude scale q95(u) from nonmissing training readings. Use the population training quantiles for an unseen household. Freeze those statistics when transforming validation data. After validation selection, refit the builder on train + validation and freeze it for test.

For observed dates belonging to household-context-slot pair (u,i), define:

- n(ui): observed count;
- f(ui) = mean[1{x > q35(u)}]: active-use frequency;
- m(ui) = min(mean(x) / max(q95(u), 10^-6), 1): bounded household-normalised magnitude;
- r(ui) = 1 / [1 + sd(x)/(mean(x) + 10^-6)]: consistency (sample standard deviation, zero when only one observation).

The implemented affinity is

    a(ui) = f(ui) [0.5 + 0.5 m(ui)] [0.5 + 0.5 r(ui)],  0 ≤ a ≤ 1.

Frequency gates the score; magnitude and repeatability modulate rather than dominate it. Saturation reduces the influence of unusually high consumption. One-observation consistency is uninformative, which is why confidence shrinks with sample count. This is a transparent heuristic measurement model with boundedness and monotonicity, not a statistically identified latent utility. Sensitivity to these internal weights is a further robustness experiment.

    c(ui) = 1 + 20 f(ui) n(ui)/(n(ui)+5).

Unobserved pairs have c=1 and implicit target p=0. This is the standard implicit-learning assumption of weak evidence for negatives, not proof that absence means dislike.

## 3. Distinct model objectives
**Surprise baselines:** observed a(ui) is used as a pseudo-rating in [0,1]. User/item KNN estimate from cosine neighbourhoods; SVD fits bias-plus-factor predictions using regularised squared error; NMF constrains factors nonnegative. Rating RMSE/MAE measure agreement with future derived affinity, not customer satisfaction.

**Implicit ALS:** use binary p(ui)=1{a(ui)≥0.5} and the confidence above, solving

    min_X,Y Σ_u,i c(ui) [p(ui) − X_uᵀY_i]² + λ (||X||²_F + ||Y||²_F).

Fixing Y, the exact user update is

    X_u = (Yᵀ C_u Y + λI)^−1 Yᵀ C_u p_u.

The code uses a linear solve, not an explicit inverse. Alternate user/item updates. The all-pairs confidence objective includes weakly weighted unobserved entries and differs from an observed-entry pseudo-rating objective. Clip predictions to [0,1] for a common ranking input; they are not calibrated probabilities.

**Context:** contextual ALS uses 96 split items. A pooled 48-slot ALS ignores weekday/weekend and maps the resulting scores back to each context. This is context-aware item splitting, not a tensor model or a learned weather/occupancy model. Popularity returns mean observed affinity by item.

## 4. Feasibility and ranking
An assumed exogenous task has original slot s0, energy E kWh over one half-hour, an allowed slot set A(u) and maximum shift Δ slots. Candidate set

    F(u) = {s ∈ A(u): |s−s0| ≤ Δ}.

The current simulator restricts to one local calendar day; it does not wrap 23:30 to 00:00. Empty sets raise a clear error. No infeasible recommendation is silently emitted.

For a 48-slot signal v, normalise N(v_s)=(v_s−min(v))/max(max(v)−min(v),10^-9). A constant signal contributes zero. Tariff and peak are normalised over the full daily signal, not separately per feasible set, preserving candidate comparability. Peak signal is the development-period mean per-reading load profile; no test consumption enters ranking. Use p̂(u,c,s) in [0,1]. Inconvenience I(u,s)=|s−s0|/max(Δ,1).

    S(u,s) = α p̂(u,c,s) − β N(price_s) − γ N(peak_s) − δ I(u,s).

Select the feasible argmax with deterministic tie-breaking. Nonnegative α,β,γ,δ express a chosen policy trade-off, not learned monetary equivalents. Default values are illustrative; one-factor-at-a-time sensitivity varies each while fixing the others. The general hybrid optimum need not be the cheapest schedule. Cost-only exhaustive enumeration is exact for this single fixed-energy task; a mixed-integer solver adds no benefit in this restricted setting.

Optional supplied carbon intensity g_s may add −ηN(g_s). It requires a real user-provided signal with provenance; there is no default fabricated carbon series or measured carbon result.

## 5. Counterfactual accounting
For complete observed baseline profile b(u,d,s), construct two augmented scenarios:

    baseline = b + E·1{s=s0};   scheduled = b + E·1{s=s*}.

The added task is hypothetical and **not extracted from b**. Both scenarios conserve energy. This avoids pretending measured aggregate demand can be arbitrarily shifted, but adding an extra task is not a reconstruction of household appliance operation.

    C = Σ_s energy_s · price_s (GBP)
    task savings = E(price_s0 − price_s*)
    total augmented bill savings (%) = 100(C_baseline−C_scheduled)/C_baseline.

Average slot power is 2·energy. Aggregate peak is max_s Σ_u 2·energy(u,d,s), not the sum of individual household maxima. Report the signed percentage difference against baseline; negative values indicate rebound. Simultaneous independent recommendations can create a new peak despite a training-profile penalty.

## 6. Evaluation and uncertainty
For top K slots R and held-out relevant slots G: Precision=|R∩G|/K, Recall=|R∩G|/|G|, HitRate=1{|R∩G|>0}; NDCG divides discounted binary gain by the ideal gain. Relevant means affinity≥0.5 using the frozen transform. Groups without relevant slots cannot define recall and are reported separately in coverage. The implementation's candidate/coverage policy is recorded in the experimental protocol.

Use chronological holdout rather than randomly scattering observations. Hyperparameters are chosen on validation; test scores never choose a model or weight. Household bootstrap first averages repeated dates and seeds within household. Resample households, preserving pairing for hybrid-versus-baseline contrasts. These intervals describe variation across the sampled synthetic households conditional on the scenario; they do not measure real-world effectiveness or account for all temporal dependence and tariff uncertainty. Real studies should additionally use temporal block resampling or repeated temporal folds.


---

# Experimental protocol

## Fixed design
The demonstration is a software verification experiment with synthetic readings. Its sample, tariff, task and constraints are explicit simulation choices. All figures and CSV measurements are generated by executing the code; none are invented benchmark values.

1. Generate 36 households × 4,032 half-hour readings with seed 11. Their stylized morning/evening peaks are mostly stationary and do not contain a deliberately modelled weekday/weekend effect. Context uplift is therefore not established by this demonstration.
2. Split unique London-local dates globally into 60% training, 20% validation and the remainder test. Timestamp ranges are in metadata. An interval-based 84-day span crosses DST and includes a partial boundary day; do not assume exactly 84 complete local dates.
3. Randomly select five household IDs from the fixed complete population with seed 11. Exclude their observations from training and validation fits. This is a fixed-population household holdout, not a forecast of newly arriving households.
4. Fit household affinity transforms on warm training only. Use population training quantiles for unknown households. Transform validation without refitting.
5. For each of seeds 11, 22, 33, fit every configuration in the explicit small grid in `src/pipeline.py`; select maximum validation NDCG with deterministic first-configuration tie-breaking.
6. Refit the affinity transform and selected models on warm train + validation. Evaluate future test affinity using frozen statistics. Implicit training and binary evaluation relevance use the same configured threshold, 0.5 by default.
7. Evaluate all observed score pairs for Surprise RMSE/MAE. Rank only household/daytype groups containing all 48 observed items and at least one relevant item; report evaluable and total context counts. Zero-positive and incomplete contexts are excluded, not converted to easy negatives. Daytype means and household means coincide for the fully eligible demo; sparse studies should inspect eligibility imbalance.
8. Store every model artifact and selected parameters. Predict unseen households with context-slot or pooled-slot training population averages, depending on the model. This deliberately conservative fallback does not solve personalised cold start.
9. Generate all feasible starts for the assumed half-hour task using externally declared availability and maximum shift. Score with contextual ALS, simulated tariff, development-period mean per-reading load profile, and inconvenience. The peak profile is a demand-shape proxy; it is not a power-flow constraint or a forecast of future feeder load.
10. Simulate only complete normal-length local test days. Add an identical 1 kWh task at the original time or proposed time. Compare no shift, cost-only exhaustive search, preference-only and hybrid under the same feasibility rules. Retain signed peak differences.
11. Run four leave-one-term-out ablations and five values (0, .25, .5, .75, 1) for each of four weights, holding the other weights fixed. Optional carbon, when supplied, stays fixed across these variants. Do not choose the best test setting and call it unbiased performance.
12. Average repeated seeds/dates within each household before bootstrap resampling. Use 400 reproducible resamples and percentile 95% intervals. Paired cost and individual-peak differences match household/date/seed before aggregation. These intervals are conditional on this synthetic scenario and do not capture cross-household feeder correlations or temporal sampling uncertainty. Aggregate peak tables are descriptive daily outcomes, not household-bootstrap intervals.

## Metrics and units
Ranking metrics are unitless. RMSE/MAE refer to affinity in [0,1]. Costs and savings are GBP per augmented household-day. `savings_percent` divides by the full augmented baseline cost; `task_savings_gbp` isolates the task delta. `shift_half_hours` converts to hours by dividing by two. Inconvenience divides the absolute shift by that household's maximum shift. Peak power is kWh / 0.5h. Aggregate peak takes the maximum after summing concurrent household profiles. Constraint violation should be zero and is verified independently against the exported household constraints.

## Unsupported-data policy
The complete combined benchmark requires nonempty warm/cold cohorts, validation and test contexts with at least one relevant item, and one complete test day. It raises clear errors before expensive fitting when these requirements fail. It does not emit fabricated zero-valued results for an undefined experiment. The preprocessing CLI remains usable for auditing insufficient data.

## Publication phase
Freeze a real-data cohort and source hashes, document treatment/tariff groups, expand validation grids, repeat seasonal temporal folds, investigate informative missingness and population selection, validate task feasibility using appliance data or participants, and include temporal/block uncertainty. Synthetic results cannot establish efficacy, novelty, demographic fairness or causal demand response.


---

# Research contribution candidates and claim boundaries

## Working research question
Can collaborative behavioural compatibility improve the trade-off between simulated task cost, demand peaks and inconvenience relative to cost-only scheduling, under explicit household constraints?

## Candidate contributions to investigate
1. A reproducible, auditable conversion of aggregate half-hour consumption into bounded behavioural affinity with observation confidence.
2. A controlled comparison of pseudo-rating CF, genuine implicit-feedback learning and context item splitting on identical temporal cohorts.
3. A constrained ranking framework that exposes separate preference, tariff, aggregate peak and inconvenience contributions.
4. A paired evaluation that separates offline behavioural recovery from counterfactual flexible-task outcomes, including synchronized-demand rebound.
5. Reproducible ablation, weight sensitivity and household-level uncertainty reporting.

These are implemented research components, **not established novel scientific contributions**. Novelty requires a systematic prior-art comparison and a convincing real-data study. A new weighted sum alone is unlikely to establish novelty. The generated demo results are software validation, not evidence of population-level efficacy.

## What would support a paper
- Pre-register primary contrasts: hybrid vs cost-only on simulated cost and inconvenience; context-aware vs pooled ALS on NDCG.
- Run several seasonal temporal windows and robust household cohorts with a documented inclusion/exclusion flow.
- Evaluate sensitivity to missingness, selection bias, tariff regime and realistic flexibility assumptions.
- Use appliance labels or participant schedules to validate availability and duration constraints.
- Report negative outcomes, rebound peaks and small/uncertain differences.
- Compare with an appropriate multi-task mixed-integer optimiser once multiple appliances are represented.
- Conduct an opt-in user study to assess recommendation acceptance and explanation usefulness.

## Explicitly not implemented
Online control, causal identification of demand response, full appliance disaggregation, electric-vehicle charging physics, solar/battery models, feeder constraints, multiple-task mixed-integer scheduling, actual LCL tariff-calendar ingestion, measured emissions reductions, deployment and user trials. Single-task cost-only exhaustive search is an exact optimum over the feasible candidate slots; it is not a general household optimiser.


---

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


---

# Sources and provenance

Sources checked on 14 November 2026. No third-party measured data are bundled.

1. UK Power Networks / London Datastore. *SmartMeter Energy Consumption Data in London Households*. https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d . Official description: 5,567 households, November 2011–February 2014, half-hourly kWh, approximately 167 million rows / 10 GB expanded. Separate 2013 dynamic tariff information is available; the package's illustrative tariff is not that calendar. Provider lists Creative Commons Attribution; inspect the downloaded terms and preserve attribution.
2. Hu, Y., Koren, Y., and Volinsky, C. (2008). *Collaborative Filtering for Implicit Feedback Datasets*. IEEE ICDM. DOI: https://doi.org/10.1109/ICDM.2008.22 . Basis of confidence-weighted implicit matrix factorisation; implementation here is a transparent small-data NumPy ALS version.
3. Surprise project and documentation. https://github.com/NicolasHug/Surprise and https://surprise.readthedocs.io/en/stable/ . Rating prediction baselines; Surprise does not support genuine implicit feedback as its general input objective. SVD denotes its regularized latent-factor rating algorithm, not direct truncated decomposition of a zero-filled matrix.
4. REFIT provider record: https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements/ . Related dataset paper DOI: https://doi.org/10.1038/sdata.2016.122 . Future extension only.
5. ACN-Data provider: https://ev.caltech.edu/dataset . Dataset paper: *ACN-Data: Analysis and Applications of an Open EV Charging Dataset*, DOI https://doi.org/10.1145/3307772.3328313 . Future extension only.
6. Ausgrid Solar Home Electricity Data provider path: https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data . Future extension only; URL availability may change, and no automatic retrieval is implemented.

## Citation discipline
Cite the original data creators in any paper. Cite actual software versions from the execution manifest. Cite this implementation as software if released, but do not invent a repository URL or DOI. No novelty, significance or real-world savings claim follows from the synthetic demonstration.
