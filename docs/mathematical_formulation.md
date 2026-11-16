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
