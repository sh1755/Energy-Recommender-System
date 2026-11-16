# Implementation plan and design decisions

The supplied user specification is the binding design. The user explicitly requested immediate implementation and execution, without stopping at a plan.

1. Implement a compact modular Python core with strict temporal boundaries, original LCL/generic CSV loading, provenance and quality audit.
2. Generate a small reproducible synthetic sample; preserve observed/missing distinction. Build context-slot interactions using train-fitted behavioural transforms.
3. Implement Surprise pseudo-rating baselines and separate confidence-weighted implicit ALS. Select small hyperparameter grids on validation only.
4. Rank feasible candidate slots with explicit exogenous tariffs and household constraints. Simulate a separate assumed flexible task rather than shifting measured total demand.
5. Execute multi-seed temporal and cold-start evaluations, paired household bootstrap, ablations and all four weight sensitivities.
6. Generate editable diagrams, result-based scientific charts, executable notebooks, methodology and Ubuntu/VS Code instructions.
7. Run tests and end-to-end demo, inspect results and visual artifacts, review scientific correctness, fix defects, package ZIP with execution evidence.

## Decisions
- Python 3.11/3.12, NumPy/Pandas, scikit-surprise and a transparent NumPy implicit-ALS implementation; no opaque API service.
- One stable daytype × half-hour item vocabulary (96 items); pooled 48-slot implicit model provides the context-free comparator.
- Offline behavioural relevance is a proxy for historical activity, not explicit preference, load flexibility or user acceptance.
- Household holdout evaluates a deliberately transparent population fallback; no cold-start personalization claim.
- Simulated price and task scenarios are not claimed to represent the actual LCL intervention.
- A monolithic full-LCL in-memory run is not the target: chunked filtering supports reproducible household/date cohorts.
- Multi-appliance scheduling, network power flow, full-scale LCL trials and online user trials remain future work.
