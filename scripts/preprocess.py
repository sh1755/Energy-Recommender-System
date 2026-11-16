"""Ingest one real dataset, validate, chronologically split and export train affinities."""
from pathlib import Path
import sys,argparse,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data import load_meter,chronological_split
from src.preprocessing import AffinityBuilder
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--config',default='config.yaml');p.add_argument('--output',default='data/processed');a=p.parse_args();c=yaml.safe_load(Path(a.config).read_text())
d=load_meter(a.input,c['chunksize'],c['timezone'],c.get('max_households'));parts=chronological_split(d,**c['split']);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
for name,frame in zip(['train','validation','test'],parts):frame.to_csv(out/f'{name}.csv',index=False)
AffinityBuilder().fit(parts[0]).transform(parts[0]).to_csv(out/'interactions.csv',index=False)
print(d.attrs.get('audit',{}))
