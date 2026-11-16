"""Independent invariants over saved experiment artifacts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];t=ROOT/'results/tables'
r=pd.read_csv(t/'recommendations.csv');e=pd.read_csv(t/'energy.csv');c=pd.read_csv(t/'household_constraints.csv');meta=json.loads((ROOT/'results/metadata.json').read_text())
assert len(pd.read_csv(t/'metrics.csv'))==8*len(meta['config']['seeds'])
parts=[x for x in r if x.endswith('_contribution')]
assert np.allclose(r.score,r[parts].sum(axis=1))
assert ((r.slot>=r.availability_start)&(r.slot<=r.availability_end)&((r.slot-r.original_slot).abs()<=r.max_shift)).all()
assert np.allclose(r.task_savings_gbp,meta['config']['task_kwh']*(r.original_tariff_gbp_kwh-r.tariff_gbp_kwh))
assert np.allclose(e.savings,e.baseline_cost-e.cost)
assert e.groupby(['seed','household_id','date']).energy_kwh.apply(lambda x:np.ptp(x.to_numpy())).max()<1e-10
joined=e.merge(c,on='household_id',validate='many_to_one')
assert ((joined.slot>=joined.availability_start)&(joined.slot<=joined.availability_end)&((joined.slot-joined.original_slot).abs()<=joined.max_shift)).all()
assert not e.constraint_violation.any()
print('PASS: model row counts, score decomposition, task costs, energy conservation and every recommendation constraint.')
# Recompute one aggregate peak directly from held-out profiles and scheduled slots.
test=pd.read_csv(ROOT/'data/processed/test.csv');row=e.iloc[0];day=e[(e.seed==row.seed)&(e.date==row.date)&(e.strategy=='hybrid')]
shifted=np.zeros(48);baseline=np.zeros(48)
for rec in day.itertuples():
    b=test[(test.household_id==rec.household_id)&(test.date==rec.date)].sort_values('slot').kwh.to_numpy()
    original=int(c.set_index('household_id').loc[rec.household_id,'original_slot'])
    bs=b.copy();ss=b.copy();bs[original]+=meta['config']['task_kwh'];ss[rec.slot]+=meta['config']['task_kwh']
    baseline+=bs;shifted+=ss
peaks=pd.read_csv(t/'aggregate_peaks.csv');p=peaks[(peaks.seed==row.seed)&(peaks.date==row.date)&(peaks.strategy=='hybrid')].iloc[0]
assert np.isclose(p.peak_kw,2*shifted.max()) and np.isclose(p.baseline_peak_kw,2*baseline.max())
print('PASS: aggregate peak independently recomputed from same-date household profiles.')
