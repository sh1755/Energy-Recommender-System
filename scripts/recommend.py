from pathlib import Path
import sys,argparse,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.recommend import recommend
p=argparse.ArgumentParser();p.add_argument('--model',default='results/models/ImplicitALS_11.joblib');p.add_argument('--household',required=True);p.add_argument('--daytype',type=int,choices=[0,1],default=0);p.add_argument('--k',type=int,default=5);a=p.parse_args()
print(json.dumps(recommend(a.model,a.household,a.daytype,a.k),indent=2))
