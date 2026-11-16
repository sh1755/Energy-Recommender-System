import numpy as np
import pandas as pd

def ranking_metrics(ranking,relevant,k=5):
    hits=np.array([i in relevant for i in ranking[:k]],dtype=float); ideal=min(k,len(relevant))
    dcg=(hits/np.log2(np.arange(len(hits))+2)).sum(); idcg=(1/np.log2(np.arange(ideal)+2)).sum()
    return {'precision':hits.sum()/k,'recall':hits.sum()/len(relevant) if relevant else 0.,'ndcg':dcg/idcg if idcg else 0.,'hit_rate':float(hits.any())}

def evaluate(model,truth,k=5,relevance_threshold=.5,ratings=False):
    rows=[]
    for (user,daytype),g in truth.assign(daytype=truth.item//48).groupby(['household_id','daytype']):
        relevant=set(g.loc[g.affinity>=relevance_threshold,'item'])
        if not relevant or len(g)!=48: continue
        items=np.arange(daytype*48,(daytype+1)*48); scores=model.predict(user,items)
        ranked=items[np.argsort(-scores,kind='stable')]; row={'household_id':user,'daytype':daytype,**ranking_metrics(ranked,relevant,k)}
        if ratings:
            err=model.predict(user,g.item.to_numpy())-g.affinity.to_numpy(); row.update(rmse=np.sqrt(np.mean(err**2)),mae=np.mean(abs(err)))
        rows.append(row)
    return pd.DataFrame(rows,columns=['household_id','daytype','precision','recall','ndcg','hit_rate']+(['rmse','mae'] if ratings else []))

def bootstrap_households(frame,columns,group_columns,seed=11,repetitions=400):
    rows=[]; rng=np.random.default_rng(seed)
    for keys,g in frame.groupby(group_columns):
        keys=keys if isinstance(keys,tuple) else (keys,)
        # Average seeds/dates first: household is the resampling unit.
        h=g.groupby('household_id')[columns].mean()
        for col in columns:
            values=h[col].dropna().to_numpy()
            if not len(values): continue
            means=np.mean(rng.choice(values,(repetitions,len(values)),replace=True),axis=1)
            rows.append(dict(zip(group_columns,keys))|{'measure':col,'mean':values.mean(),'ci_low':np.quantile(means,.025),'ci_high':np.quantile(means,.975),'households':len(values)})
    return pd.DataFrame(rows)
