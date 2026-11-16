"""Chunked ingestion. Naive LCL clock readings are interpreted as Europe/London.
Ambiguous/nonexistent DST times are rejected, never guessed. Exact duplicate
readings are collapsed; conflicting duplicates reject. Missing kWh stays flagged.
"""
from pathlib import Path
import numpy as np
import pandas as pd

def enrich(d):
    d=d.copy(); local=d.timestamp.dt.tz_convert('Europe/London')
    d['date']=local.dt.date.astype(str); d['daytype']=(local.dt.dayofweek>=5).astype(int)
    d['slot']=local.dt.hour*2+local.dt.minute//30; d['item']=d.daytype*48+d.slot
    d['missing']=d.kwh.isna()
    return d.sort_values(['timestamp','household_id']).reset_index(drop=True)

def load_meter(path, chunksize=250000, timezone='Europe/London', max_households=None):
    files=sorted(Path(path).glob('*.csv')) if Path(path).is_dir() else [Path(path)]
    if not files: raise ValueError('No CSV files found')
    chunks=[]; selected=set()
    for file in files:
        for c in pd.read_csv(file,chunksize=chunksize,dtype=str):
            c.columns=c.columns.str.strip(); c=c.rename(columns={'LCLid':'household_id','DateTime':'timestamp','KWH/hh (per half hour)':'kwh'})
            if not {'household_id','timestamp','kwh'}<=set(c): raise ValueError('Required columns: household_id,timestamp,kwh or LCL names')
            c=c[['household_id','timestamp','kwh']].copy()
            if c.household_id.isna().any(): raise ValueError('Missing household identifier')
            c.household_id=c.household_id.astype(str).str.strip()
            if c.household_id.eq('').any(): raise ValueError('Missing household identifier')
            if max_households:
                for u in c.household_id.unique():
                    if len(selected)<max_households: selected.add(u)
                c=c[c.household_id.isin(selected)].copy()
            if c.empty: continue
            raw=c.kwh.astype(str).str.strip(); c['kwh']=pd.to_numeric(raw,errors='coerce')
            invalid=c.kwh.isna() & ~raw.str.lower().isin(['null','nan','','na','none'])
            if invalid.any(): raise ValueError('Non-numeric kWh values')
            if (c.kwh.dropna()<0).any() or np.isinf(c.kwh.dropna()).any(): raise ValueError('kWh must be finite and nonnegative')
            ts=pd.to_datetime(c.timestamp,format='mixed',errors='raise')
            if ts.isna().any(): raise ValueError('Missing timestamp')
            if ts.dtype == object:
                if not all(getattr(t,'tzinfo',None) is not None for t in ts): raise ValueError('Mixed naive and aware timestamps; supply explicit offsets consistently')
                ts=pd.to_datetime(c.timestamp,format='mixed',utc=True,errors='raise')
            if ts.dt.tz is None: ts=ts.dt.tz_localize(timezone,ambiguous='raise',nonexistent='raise')
            c['timestamp']=ts.dt.tz_convert('UTC')
            if (c.timestamp != c.timestamp.dt.floor('30min')).any(): raise ValueError('Readings must align to half hours')
            chunks.append(c)
    if not chunks: raise ValueError('No observations')
    d=pd.concat(chunks,ignore_index=True)
    if (d.groupby(['household_id','timestamp']).kwh.nunique(dropna=False)>1).any(): raise ValueError('conflicting duplicate household/timestamp readings')
    duplicates=int(d.duplicated(['household_id','timestamp']).sum())
    result=enrich(d.drop_duplicates(['household_id','timestamp']))
    expected=sum(int((g.timestamp.max()-g.timestamp.min()).total_seconds()/1800)+1 for _,g in result.groupby('household_id'))
    result.attrs['audit']={'exact_duplicates_removed':duplicates,'missing_value_rows':int(result.missing.sum()),'absent_halfhour_intervals':expected-len(result),'selected_households':int(result.household_id.nunique())}
    return result

def synthetic_data(households=36,days=84,seed=11):
    rng=np.random.default_rng(seed); ts=pd.date_range('2024-01-08',periods=days*48,freq='30min',tz='Europe/London')
    slot=ts.hour.to_numpy()*2+ts.minute.to_numpy()/30; records=[]
    for u in range(households):
        morning=rng.uniform(12,17); evening=rng.uniform(33,42)
        profile=.08+rng.uniform(.1,.4)*np.exp(-((slot-morning)/3)**2)+rng.uniform(.3,.9)*np.exp(-((slot-evening)/4)**2)
        kwh=np.maximum(0,profile*rng.lognormal(0,.3,len(ts))+rng.normal(0,.015,len(ts)))
        records.append(pd.DataFrame({'household_id':f'H{u:04d}','timestamp':ts.tz_convert('UTC'),'kwh':kwh}))
    return enrich(pd.concat(records,ignore_index=True))

def chronological_split(d,train_fraction=.6,validation_fraction=.2):
    dates=sorted(d.date.unique()); n=len(dates); i=int(n*train_fraction); j=int(n*(train_fraction+validation_fraction))
    if not 1<=i<j<n: raise ValueError('Need enough dates for three nonempty chronological splits')
    return tuple(d[d.date.isin(part)].copy() for part in (dates[:i],dates[i:j],dates[j:]))
