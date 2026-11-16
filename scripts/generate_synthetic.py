from pathlib import Path
import sys,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data import synthetic_data
p=argparse.ArgumentParser(); p.add_argument('--output',default='data/raw/synthetic.csv'); p.add_argument('--households',type=int,default=36); p.add_argument('--days',type=int,default=84); p.add_argument('--seed',type=int,default=11); a=p.parse_args()
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
synthetic_data(a.households,a.days,a.seed)[['household_id','timestamp','kwh']].to_csv(a.output,index=False)
