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
