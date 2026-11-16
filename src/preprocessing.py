import numpy as np
class AffinityBuilder:
    """Household q95 scales and q35 activity thresholds fitted only on history.
    Cold households use training-global fallback. Missing is excluded, zero observed.
    """
    def fit(self,d):
        v=d.loc[~d.missing]; self.scale=max(float(v.kwh.quantile(.95)),1e-6); self.threshold=float(v.kwh.quantile(.35))
        self.scales=v.groupby('household_id').kwh.quantile(.95).clip(lower=1e-6).to_dict(); self.thresholds=v.groupby('household_id').kwh.quantile(.35).to_dict(); return self
    def transform(self,d):
        v=d.loc[~d.missing].copy(); thresholds=v.household_id.map(self.thresholds).fillna(self.threshold)
        v['active']=(v.kwh>thresholds).astype(float)
        g=v.groupby(['household_id','item']).agg(mean_kwh=('kwh','mean'),std_kwh=('kwh','std'),frequency=('active','mean'),observations=('kwh','size')).reset_index()
        g['consistency']=1/(1+g.std_kwh.fillna(0)/(g.mean_kwh+1e-6))
        scales=g.household_id.map(self.scales).fillna(self.scale)
        g['affinity']=(g.frequency*(.5+.5*np.minimum(g.mean_kwh/scales,1))*(.5+.5*g.consistency)).clip(0,1)
        g['confidence']=1+20*g.frequency*g.observations/(g.observations+5)
        return g
