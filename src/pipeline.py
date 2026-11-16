"""Run a complete experiment. Test outcomes never select model hyperparameters."""
import argparse,json,platform
from pathlib import Path
from importlib.metadata import version
import joblib,numpy as np,pandas as pd,yaml
from .data import load_meter,synthetic_data,chronological_split
from .preprocessing import AffinityBuilder
from .models import make_model
from .evaluation import evaluate,bootstrap_households
from .energy import simulated_tariff,hybrid_scores,simulate_day,load_carbon,household_constraints

GRIDS={'Popularity':[{}],'PooledPopularity':[{}],'UserKNN':[{'k':10},{'k':20}],'ItemKNN':[{'k':10},{'k':20}],'SVD':[{'n_factors':8,'n_epochs':30},{'n_factors':16,'n_epochs':30}],'NMF':[{'n_factors':8,'n_epochs':30},{'n_factors':16,'n_epochs':30}],'PooledImplicitALS':[{'factors':8,'regularization':.1},{'factors':12,'regularization':.5}],'ImplicitALS':[{'factors':8,'regularization':.1},{'factors':12,'regularization':.5}]}

def validate_config(config):
    def finite(value):
        return isinstance(value,(int,float)) and not isinstance(value,bool) and np.isfinite(value)
    def positive_integer(value):
        return finite(value) and int(value)==value and value>0
    if not finite(config.get('task_kwh')) or config['task_kwh']<=0: raise ValueError('task_kwh must be finite and positive')
    if not positive_integer(config.get('k')) or config['k']>48: raise ValueError('k must be an integer from 1 to 48')
    seeds=config.get('seeds')
    if not isinstance(seeds,list) or not seeds or any(not finite(s) or int(s)!=s or s<0 for s in seeds) or len(set(seeds))!=len(seeds): raise ValueError('seeds must be a nonempty list of distinct nonnegative integers')
    weights=config.get('weights')
    if not isinstance(weights,list) or len(weights)!=4 or any(not finite(w) or w<0 for w in weights): raise ValueError('weights must have four finite nonnegative values')
    fractions=config.get('split',{}); tf=fractions.get('train_fraction'); vf=fractions.get('validation_fraction')
    if not finite(tf) or not finite(vf) or tf<=0 or vf<=0 or tf+vf>=1: raise ValueError('Split fractions must be positive with sum below one')
    cf=config.get('cold_start_fraction')
    if not finite(cf) or not 0<cf<1: raise ValueError('cold_start_fraction must lie strictly between zero and one')
    if not finite(config.get('relevance_threshold')) or not 0<=config['relevance_threshold']<=1: raise ValueError('relevance_threshold must be within [0,1]')
    if not positive_integer(config.get('bootstrap_repetitions')): raise ValueError('bootstrap_repetitions must be a positive integer')
    if not finite(config.get('carbon_weight',0)) or config.get('carbon_weight',0)<0: raise ValueError('carbon_weight must be finite and nonnegative')
    availability=config.get('availability')
    if not isinstance(availability,list) or len(availability)!=2: raise ValueError('availability requires two integer slots')
    household_constraints(['__validation__'],{**config,'constraints_csv':None})

def require_evaluable(frame,threshold,label):
    groups=frame.assign(daytype=frame.item//48).groupby(['household_id','daytype'])
    if not any(len(g)==48 and (g.affinity>=threshold).any() for _,g in groups):
        raise ValueError(f'{label} has no evaluable 48-slot household contexts with relevant items; supply more observed history or revise the declared relevance threshold')

def run(config,input_path=None):
    validate_config(config)
    out=Path(config.get('output_dir','results')); tables=out/'tables'; models_dir=out/'models'; processed=Path('data/processed')
    for p in [tables,models_dir,processed]: p.mkdir(parents=True,exist_ok=True)
    if input_path: data=load_meter(input_path,config['chunksize'],config['timezone'],config.get('max_households'))
    else: data=synthetic_data(**config['synthetic'],seed=config['seed'])
    if data.household_id.nunique()<2: raise ValueError('At least two households are required for warm and cold cohorts')
    if not input_path:
        Path('data/synthetic').mkdir(parents=True,exist_ok=True)
        data[['household_id','timestamp','kwh']].to_csv('data/synthetic/synthetic_meter.csv',index=False)
    audit=data.attrs.get('audit',{'missing_value_rows':int(data.missing.sum()),'source':'synthetic generator'})
    train,val,test=chronological_split(data,**config['split'])
    users=np.array(sorted(data.household_id.unique())); rng=np.random.default_rng(config['seed']); cold=set(rng.choice(users,max(1,int(len(users)*config['cold_start_fraction'])),replace=False))
    train_w=train[~train.household_id.isin(cold)]; val_w=val[~val.household_id.isin(cold)]
    if train_w.loc[~train_w.missing].empty or val_w.loc[~val_w.missing].empty: raise ValueError('Warm training and validation cohorts require observed readings')
    builder=AffinityBuilder().fit(train_w); a=builder.transform(train_w); v=builder.transform(val_w)
    require_evaluable(v,config['relevance_threshold'],'Validation')
    for name,frame in [('train',train),('validation',val),('test',test)]: frame.to_csv(processed/f'{name}.csv',index=False)
    a.to_csv(processed/'interactions.csv',index=False)
    data.groupby('household_id').agg(rows=('kwh','size'),observed=('kwh','count'),missing=('missing','sum')).to_csv(tables/'data_quality.csv')
    fit_data=pd.concat([train_w,val_w]); final_builder=AffinityBuilder().fit(fit_data); final_a=final_builder.transform(fit_data)
    truth=final_builder.transform(test[~test.household_id.isin(cold)]); cold_truth=final_builder.transform(test[test.household_id.isin(cold)])
    require_evaluable(truth,config['relevance_threshold'],'Warm test')
    require_evaluable(cold_truth,config['relevance_threshold'],'Cold test')
    tariff=simulated_tariff(); carbon=None
    if config.get('carbon_csv'):
        if not config.get('carbon_provenance'): raise ValueError('Provide carbon_provenance for supplied carbon scenario')
        carbon=load_carbon(config['carbon_csv'])
    original=config['original_slot']; max_shift=config['max_shift']; lo,hi=config['availability']
    candidates=np.array([s for s in range(48) if abs(s-original)<=max_shift and lo<=s<=hi])
    if not len(candidates) or original not in candidates: raise ValueError('Availability must include original slot and feasible candidates')
    constraints=household_constraints(users,config)
    pd.DataFrame([{'household_id':u,**c} for u,c in constraints.items()]).to_csv(tables/'household_constraints.csv',index=False)
    peak_profiles=fit_data[~fit_data.missing].groupby(['daytype','slot']).kwh.mean()
    tuning=[]; metrics=[]; households=[]; cold_rows=[]; recommendations=[]; energy=[]; variants=[]; aggregate=[]
    # Only complete normal-length local days can support counterfactual half-hour accounting.
    complete=[]
    for (user,date),g in test.groupby(['household_id','date']):
        if len(g)!=48 or g.slot.nunique()!=48 or g.missing.any(): continue
        complete.append((user,date,int(g.daytype.iloc[0]),g.sort_values('slot').kwh.to_numpy()))
    if not complete: raise ValueError('Test data have no complete 48-slot household-days for simulation; supply complete local days')
    for seed in config['seeds']:
        fitted={}
        for name,grid in GRIDS.items():
            best=(-np.inf,None)
            for params in grid:
                params=dict(params)
                if name in ['ImplicitALS','PooledImplicitALS']: params['preference_threshold']=config['relevance_threshold']
                model=make_model(name,seed,params).fit(a); ev=evaluate(model,v,config['k'],config['relevance_threshold'])
                score=float(ev.ndcg.mean()) if len(ev) else 0
                tuning.append({'seed':seed,'model':name,'params':json.dumps(params),'validation_ndcg':score})
                if score>best[0]: best=(score,params)
            model=make_model(name,seed,best[1]).fit(final_a); fitted[name]=model
            joblib.dump({'model':model,'builder':final_builder,'params':best[1],'seed':seed},models_dir/f'{name}_{seed}.joblib')
            ev=evaluate(model,truth,config['k'],config['relevance_threshold'],name in ['UserKNN','ItemKNN','SVD','NMF'])
            ev['seed']=seed; ev['model']=name; households.append(ev)
            means=ev.select_dtypes('number').drop(columns=['daytype','seed'],errors='ignore').mean().to_dict()
            
            if name in ['UserKNN','ItemKNN','SVD','NMF']:
                errors=np.concatenate([model.predict(u,g.item.to_numpy())-g.affinity.to_numpy() for u,g in truth.groupby('household_id')])
                means.update(rmse=float(np.sqrt(np.mean(errors**2))),mae=float(np.mean(abs(errors))))
            metrics.append({'total_household_contexts':len(truth.assign(daytype=truth.item//48).groupby(['household_id','daytype'])),'seed':seed,'model':name,'k':config['k'],'evaluable_household_contexts':len(ev),**means})
            ce=evaluate(model,cold_truth,config['k'],config['relevance_threshold']); ce['seed']=seed; ce['model']=name; ce['fallback']='training pooled-slot popularity' if name.startswith('Pooled') else 'training context-slot popularity'; cold_rows.append(ce)
        als=fitted['ImplicitALS']; ranking_cache={}
        weights=list(config['weights']); settings={'hybrid':weights,'preference_only':[1,0,0,0],'cost_only':[0,1,0,0],'no_shift':[0,0,0,0]}
        for i,label in enumerate(['alpha','beta','gamma','delta']):
            w=weights.copy(); w[i]=0; settings[f'ablation_without_{label}']=w
            for level in [0.,.25,.5,.75,1.]:
                w=weights.copy(); w[i]=level; settings[f'sensitivity_{label}_{level}']=w
        for user in users:
            constraint=constraints[user]; original=constraint['original_slot']; max_shift=constraint['max_shift']
            lo=constraint['availability_start']; hi=constraint['availability_end']
            candidates=np.array([s for s in range(48) if abs(s-original)<=max_shift and lo<=s<=hi])
            for daytype in [0,1]:
                pref=als.predict(user,np.arange(daytype*48,(daytype+1)*48)); peak=peak_profiles.reindex([(daytype,s) for s in range(48)]).fillna(0).to_numpy()
                for strategy,w in settings.items():
                    score,parts=hybrid_scores(pref,tariff,peak,original,max_shift,w,carbon,config.get('carbon_weight',0) if strategy=='hybrid' or strategy.startswith(('ablation','sensitivity')) else 0)
                    slot=original if strategy=='no_shift' else int(candidates[np.argmax(score[candidates])]); ranking_cache[user,daytype,strategy]=slot
                    if strategy=='hybrid':
                        for rank,s in enumerate(candidates[np.argsort(-score[candidates],kind='stable')][:config['k']],1):
                            contributions={k:float(vv[s]) for k,vv in parts.items()}
                            explanation=f"Recommend {int(s)//2:02d}:{(int(s)%2)*30:02d}: tariff £{tariff[s]:.2f}/kWh vs £{tariff[original]:.2f} at original time; feasible within availability slots {lo}–{hi} and shift limit {max_shift} half-hours; learned affinity {pref[s]:.3f}. "+'; '.join(f'{k}={vv:+.6f}' for k,vv in contributions.items())+'; score=sum(contributions); aggregate-meter affinity, not appliance preference'
                            recommendations.append({'seed':seed,'household_id':user,'cold_start':user in cold,'daytype':daytype,'rank':rank,'item':daytype*48+int(s),'slot':int(s),'score':float(score[s]),'tariff_gbp_kwh':float(tariff[s]),'original_tariff_gbp_kwh':float(tariff[original]),'task_savings_gbp':float(config['task_kwh']*(tariff[original]-tariff[s])),'original_slot':original,'max_shift':max_shift,'availability_start':lo,'availability_end':hi,'constraint_violation':False,**contributions,'explanation':explanation})
        daily_profiles={}
        for user,date,daytype,base in complete:
            constraint=constraints[user]; original=constraint['original_slot']; max_shift=constraint['max_shift']
            for strategy in settings:
                slot=ranking_cache[user,daytype,strategy]; result=simulate_day(base,tariff,slot,original,config['task_kwh'],carbon)
                row={'seed':seed,'strategy':strategy,'household_id':user,'date':date,'slot':slot,'shift_half_hours':abs(slot-original),'inconvenience':abs(slot-original)/max(max_shift,1),'constraint_violation':not (constraint['availability_start']<=slot<=constraint['availability_end'] and abs(slot-original)<=max_shift),**result}
                if strategy.startswith(('ablation','sensitivity')): variants.append(row)
                else: energy.append(row)
                key=(date,strategy)
                if key not in daily_profiles: daily_profiles[key]=[np.zeros(48),np.zeros(48),0]
                shifted=base.copy(); shifted[slot]+=config['task_kwh']; baseline=base.copy(); baseline[original]+=config['task_kwh']
                daily_profiles[key][0]+=shifted; daily_profiles[key][1]+=baseline; daily_profiles[key][2]+=1
        for (date,strategy),(shifted,baseline,n) in daily_profiles.items(): aggregate.append({'seed':seed,'date':date,'strategy':strategy,'households':n,'peak_kw':2*shifted.max(),'baseline_peak_kw':2*baseline.max(),'peak_reduction_kw':2*(baseline.max()-shifted.max()),'peak_reduction_percent':100*(baseline.max()-shifted.max())/baseline.max() if baseline.max() else 0})
        print(f'Seed {seed}: models and simulations complete',flush=True)
    metric_df=pd.DataFrame(metrics); hh=pd.concat(households,ignore_index=True); en=pd.DataFrame(energy); var=pd.DataFrame(variants)
    exports={'metrics':metric_df,'household_metrics':hh,'tuning':pd.DataFrame(tuning),'cold_start':pd.concat(cold_rows,ignore_index=True),'recommendations':pd.DataFrame(recommendations),'energy':en,'ablation':var[var.strategy.str.startswith('ablation')],'sensitivity':var[var.strategy.str.startswith('sensitivity')],'aggregate_peaks':pd.DataFrame(aggregate)}
    uncertainty=bootstrap_households(hh,['precision','recall','ndcg','hit_rate'],['model'],repetitions=config['bootstrap_repetitions']); uncertainty['type']='ranking'
    eu=bootstrap_households(en,['savings','peak_reduction_kw','shift_half_hours'],['strategy'],repetitions=config['bootstrap_repetitions']); eu['type']='energy'; exports['uncertainty']=pd.concat([uncertainty,eu],ignore_index=True)
    pairs=[]
    h=en[en.strategy=='hybrid']
    for comp in ['no_shift','cost_only','preference_only']:
        merged=h.merge(en[en.strategy==comp],on=['seed','household_id','date'],suffixes=('_hybrid','_comparator'))
        for metric in ['cost','peak_kw']:
            # Positive means hybrid reduces this quantity relative to comparator.
            p=merged[['household_id']].copy(); p['difference']=merged[f'{metric}_comparator']-merged[f'{metric}_hybrid']; p['comparison']=f'hybrid_vs_{comp}'; p['metric']=metric
            pairs.append(p)
    exports['paired_comparisons']=bootstrap_households(pd.concat(pairs),['difference'],['comparison','metric'],repetitions=config['bootstrap_repetitions'])
    exports['across_seed_summary']=metric_df.groupby('model')[['precision','recall','ndcg','hit_rate']].agg(['mean','std']).reset_index()
    exports['across_seed_summary'].columns=['_'.join(c).rstrip('_') if isinstance(c,tuple) else c for c in exports['across_seed_summary'].columns]
    for name,frame in exports.items(): frame.to_csv(tables/f'{name}.csv',index=False)
    metadata={'data_audit':audit,'data_source':str(input_path) if input_path else 'synthetic demonstration; no measured outcomes','tariff':'simulated GBP/kWh: 0.14 before 07:00, 0.40 16:00–20:00, otherwise 0.24','carbon_provenance':config.get('carbon_provenance'),'config':config,'cold_households':sorted(cold),'split_date_ranges':{n:[str(d.timestamp.min()),str(d.timestamp.max())] for n,d in [('train',train),('validation',val),('test',test)]},'rows':len(data),'complete_test_household_days':len(complete),'test_household_days':len(test.groupby(['household_id','date'])),'python':platform.python_version(),'versions':{p:version(p) for p in ['numpy','pandas','scipy','scikit-surprise','joblib']},'methodology':'Validation selects hyperparameters. Refit train+validation; one final test. Household bootstrap averages seeds first; no pseudoreplication. Affinities are aggregate-meter proxies. Energy task is exogenous and added to observed baseline. Signed aggregate peaks include rebound.'}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)); return exports

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',default='config.yaml'); parser.add_argument('--input'); args=parser.parse_args()
    run(yaml.safe_load(Path(args.config).read_text()),args.input)
if __name__=='__main__': main()
