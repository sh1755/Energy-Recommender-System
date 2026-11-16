"""Scenario accounting, not a causal estimate of appliance-level savings."""
import numpy as np
import pandas as pd

def simulated_tariff():
    t=np.full(48,.24); t[:14]=.14; t[32:40]=.40; return t

def hybrid_scores(preference,tariff,peak,original_slot,max_shift,weights,carbon=None,carbon_weight=0):
    alpha,beta,gamma,delta=weights
    cost=(tariff-tariff.min())/max(np.ptp(tariff),1e-9)
    peak=(peak-peak.min())/max(np.ptp(peak),1e-9)
    discomfort=np.abs(np.arange(48)-original_slot)/max(max_shift,1)
    parts={'preference_contribution':alpha*np.clip(preference,0,1),'cost_contribution':-beta*cost,'peak_contribution':-gamma*peak,'comfort_contribution':-delta*discomfort}
    if carbon is not None: parts['carbon_contribution']=-carbon_weight*(carbon-carbon.min())/max(np.ptp(carbon),1e-9)
    return sum(parts.values()),parts

def simulate_day(base,tariff,slot,original_slot=36,task_kwh=1,carbon=None):
    base=np.asarray(base,dtype=float)
    if base.shape!=(48,) or not np.isfinite(base).all() or (base<0).any(): raise ValueError('Simulation requires complete nonnegative 48-slot observed day')
    tariff=np.asarray(tariff,dtype=float)
    if tariff.shape!=(48,) or not np.isfinite(tariff).all() or (tariff<0).any(): raise ValueError('Tariff must contain 48 finite nonnegative values')
    if not np.isfinite(task_kwh) or task_kwh<=0: raise ValueError('task_kwh must be finite and positive')
    if any(isinstance(s,bool) or not isinstance(s,(int,np.integer)) or not 0<=s<48 for s in [slot,original_slot]): raise ValueError('Task slots must be integers from 0 to 47')
    if carbon is not None:
        carbon=np.asarray(carbon,dtype=float)
        if carbon.shape!=(48,) or not np.isfinite(carbon).all() or (carbon<0).any(): raise ValueError('Carbon must contain 48 finite nonnegative values')
    shifted=base.copy(); baseline=base.copy(); shifted[slot]+=task_kwh; baseline[original_slot]+=task_kwh
    cost=float(shifted@tariff); baseline_cost=float(baseline@tariff)
    out={'cost':cost,'baseline_cost':baseline_cost,'savings':baseline_cost-cost,'energy_kwh':shifted.sum(),'peak_kw':shifted.max()*2,'baseline_peak_kw':baseline.max()*2,'peak_reduction_kw':(baseline.max()-shifted.max())*2}
    out['savings_percent']=100*out['savings']/baseline_cost if baseline_cost else 0.
    out['peak_reduction_percent']=100*out['peak_reduction_kw']/out['baseline_peak_kw'] if out['baseline_peak_kw'] else 0.
    if carbon is not None: out.update(carbon_g=float(shifted@carbon),baseline_carbon_g=float(baseline@carbon),carbon_reduction_g=float((baseline-shifted)@carbon))
    return out

def load_carbon(path):
    c=pd.read_csv(path)
    if not {'slot','gco2_per_kwh'}<=set(c) or len(c)!=48 or set(c.slot)!=set(range(48)): raise ValueError('Carbon CSV requires exactly slots 0..47 and gco2_per_kwh')
    v=c.sort_values('slot').gco2_per_kwh.to_numpy(float)
    if not np.isfinite(v).all() or (v<0).any(): raise ValueError('Invalid carbon values')
    return v


def household_constraints(users,config):
    def integer(value):
        if isinstance(value,bool) or not np.isfinite(value) or int(value)!=value: raise ValueError('Constraint slots and shifts must be finite integer values')
        return int(value)
    defaults={u:{'original_slot':integer(config['original_slot']),'max_shift':integer(config['max_shift']),'availability_start':integer(config['availability'][0]),'availability_end':integer(config['availability'][1])} for u in users}
    if config.get('constraints_csv'):
        frame=pd.read_csv(config['constraints_csv'],dtype={'household_id':str})
        required={'household_id','original_slot','max_shift','availability_start','availability_end'}
        if not required<=set(frame) or frame.household_id.duplicated().any(): raise ValueError('Invalid household constraints CSV')
        for row in frame.to_dict('records'):
            if row['household_id'] in defaults: defaults[row['household_id']]={k:integer(row[k]) for k in required-{'household_id'}}
    for u,c in defaults.items():
        if not 0<=c['availability_start']<=c['original_slot']<=c['availability_end']<48 or not 0<=c['max_shift']<=47: raise ValueError(f'Invalid constraints for {u}')
    return defaults
