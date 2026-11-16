import numpy as np
import pandas as pd
import pytest
from src.data import load_meter, synthetic_data, chronological_split
from src.preprocessing import AffinityBuilder
from src.models import ImplicitALS
from src.energy import simulate_day, hybrid_scores
from src.evaluation import ranking_metrics

def test_conflicting_duplicates(tmp_path):
    p=tmp_path/'x.csv'
    pd.DataFrame({'household_id':['a','a'],'timestamp':['2024-01-01','2024-01-01'],'kwh':[1,2]}).to_csv(p,index=False)
    with pytest.raises(ValueError,match='conflicting'): load_meter(p)

def test_missing_and_split(tmp_path):
    p=tmp_path/'x.csv'
    pd.DataFrame({'LCLid':['a','a'],'DateTime':['2024-01-01','2024-01-01 00:30'],'KWH/hh (per half hour) ':['Null','1']}).to_csv(p,index=False)
    d=load_meter(p); assert d.missing.sum()==1
    a,b,c=chronological_split(synthetic_data(4,20,11))
    assert a.timestamp.max()<b.timestamp.min()<c.timestamp.min()

def test_train_only_transform():
    d=synthetic_data(4,20,11); a,b,c=chronological_split(d)
    f=AffinityBuilder().fit(a); bound=f.scale
    x=f.transform(c.assign(kwh=1000))
    assert f.scale==bound and x.affinity.between(0,1).all()

def test_als_and_cold():
    a=pd.DataFrame({'household_id':['a','a','b'],'item':[0,1,1],'affinity':[.8,.2,.9],'confidence':[4,2,5]})
    m=ImplicitALS(factors=3,iterations=3,seed=11).fit(a)
    assert np.isfinite(m.predict('a',np.arange(96))).all()
    assert np.isfinite(m.predict('new',np.arange(96))).all()

def test_exogenous_energy_and_signed_peak():
    base=np.ones(48)*.2; tariff=np.ones(48)*.2
    out=simulate_day(base,tariff,0,36,1)
    assert out['energy_kwh']==pytest.approx(base.sum()+1)
    assert out['savings']==pytest.approx(0)
    assert out['peak_reduction_kw']==pytest.approx(0)

def test_ranking_known():
    m=ranking_metrics([2,1,3],{1,2},2)
    assert m['precision']==m['recall']==m['ndcg']==m['hit_rate']==1

def test_score_accounting():
    score,parts=hybrid_scores(np.arange(48)/48,np.ones(48)*.2,np.arange(48)/48,36,8,[.4,.3,.2,.1])
    assert np.allclose(score,sum(parts.values()))

def test_household_scale_invariance():
    d=synthetic_data(2,20,11); doubled=d.copy(); doubled.loc[doubled.household_id=='H0000','kwh']*=10
    original=AffinityBuilder().fit(d).transform(d)
    scaled=AffinityBuilder().fit(doubled).transform(doubled)
    assert np.allclose(original.affinity,scaled.affinity,atol=1e-5)

def test_missing_context_not_ranked():
    from src.evaluation import evaluate
    from src.models import Popularity
    a=pd.DataFrame({'household_id':['a'],'item':[0],'affinity':[1.]})
    assert evaluate(Popularity().fit(a),a).empty

def test_model_roundtrip(tmp_path):
    import joblib
    from src.recommend import recommend
    from src.models import Popularity
    a=AffinityBuilder().fit(synthetic_data(2,10)).transform(synthetic_data(2,10))
    path=tmp_path/'model.joblib'; joblib.dump({'model':Popularity().fit(a)},path)
    assert len(recommend(path,'new',1,5))==5

def test_utc_dst_input_supported(tmp_path):
    p=tmp_path/'utc.csv'; pd.DataFrame({'household_id':['a','a'],'timestamp':['2024-10-27 01:00:00+00:00','2024-10-27 01:30:00+00:00'],'kwh':[1,2]}).to_csv(p,index=False)
    assert len(load_meter(p))==2

def test_household_constraint_override(tmp_path):
    from src.energy import household_constraints
    p=tmp_path/'constraints.csv'
    pd.DataFrame([{'household_id':'a','original_slot':20,'max_shift':2,'availability_start':18,'availability_end':23}]).to_csv(p,index=False)
    c=household_constraints(['a','b'],{'original_slot':36,'max_shift':8,'availability':[28,44],'constraints_csv':str(p)})
    assert c['a']['original_slot']==20 and c['b']['original_slot']==36

def test_absent_interval_audit(tmp_path):
    p=tmp_path/'x.csv'; pd.DataFrame({'household_id':['a','a','a'],'timestamp':['2024-01-01 00:00','2024-01-01 01:00','2024-01-01 01:00'],'kwh':[1,2,2]}).to_csv(p,index=False)
    d=load_meter(p,chunksize=1)
    assert d.attrs['audit']['exact_duplicates_removed']==1
    assert d.attrs['audit']['absent_halfhour_intervals']==1

def test_config_rejects_invalid_controls():
    import yaml
    from src.pipeline import validate_config
    from pathlib import Path
    cfg=yaml.safe_load(Path('config.yaml').read_text())
    for key,value in [('task_kwh',-1),('task_kwh',float('nan')),('k',0),('k',2.5),('seeds',[]),('weights',[.1,.2,-1,.4]),('availability',[28.5,44])]:
        bad=dict(cfg); bad[key]=value
        with pytest.raises(ValueError): validate_config(bad)

def test_pipeline_rejects_all_cold(tmp_path):
    import yaml
    from pathlib import Path
    from src.pipeline import run
    cfg=yaml.safe_load(Path('config.yaml').read_text());cfg['synthetic']={'households':1,'days':20};cfg['output_dir']=str(tmp_path/'out')
    with pytest.raises(ValueError,match='households'):run(cfg)

def test_constraint_fractional_rejected():
    from src.energy import household_constraints
    with pytest.raises(ValueError,match='integer'):
        household_constraints(['a'],{'original_slot':36.5,'max_shift':8,'availability':[28,44]})

def test_ids_and_missing_timestamp(tmp_path):
    p=tmp_path/'x.csv'
    p.write_text('household_id,timestamp,kwh\n001,2024-01-01 00:00,1\n')
    assert load_meter(p).household_id.iloc[0]=='001'
    p.write_text('household_id,timestamp,kwh\n001,,1\n')
    with pytest.raises(ValueError,match='timestamp'): load_meter(p)

def test_subsecond_timestamp_rejected(tmp_path):
    p=tmp_path/'x.csv';p.write_text('household_id,timestamp,kwh\na,2024-01-01 00:00:00.001,1\n')
    with pytest.raises(ValueError,match='half hours'): load_meter(p)

@pytest.mark.parametrize('energy',[-1,float('nan'),float('inf')])
def test_unphysical_task_rejected(energy):
    with pytest.raises(ValueError,match='task_kwh'): simulate_day(np.ones(48),np.ones(48),20,task_kwh=energy)

def test_empty_evaluation_diagnostic():
    from src.pipeline import require_evaluable
    a=pd.DataFrame({'household_id':['a'],'item':[0],'affinity':[0.]})
    with pytest.raises(ValueError,match='no evaluable'): require_evaluable(a,.5,'Validation')

def test_implicit_target_threshold():
    a=pd.DataFrame({'household_id':['a'],'item':[0],'affinity':[.6],'confidence':[5]})
    low=ImplicitALS(factors=2,iterations=3,preference_threshold=.5).fit(a)
    high=ImplicitALS(factors=2,iterations=3,preference_threshold=.8).fit(a)
    assert low.predict('a',[0])[0]>0
    assert high.predict('a',[0])[0]==0
