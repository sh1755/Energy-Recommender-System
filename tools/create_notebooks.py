"""Create lightweight, executable tutorials backed by actual project modules."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[1]
setup="""from pathlib import Path
import sys
ROOT = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
"""
books=[
('01_dataset_exploration','Dataset exploration','All bundled readings are synthetic. The original LCL schema is supported by src.data.load_meter. Missing readings are never silently replaced by zero.',[
"train = pd.read_csv(ROOT/'data/processed/train.csv')\nprint(train.shape)\ntrain.head()",
"train.groupby('slot').kwh.mean().plot(xlabel='Half-hour slot', ylabel='Mean kWh', title='Synthetic training profile');",
"pd.read_csv(ROOT/'results/tables/data_quality.csv')"]),
('02_interaction_generation','Interaction generation','Fit on training readings only. Affinity measures historical activity compatibility, not explicit preference or flexibility.',[
"from src.data import load_meter\nfrom src.preprocessing import AffinityBuilder\ntrain = load_meter(ROOT/'data/processed/train.csv')\nbuilder = AffinityBuilder().fit(train)\ninteractions = builder.transform(train)\ninteractions.head()",
"matrix = interactions.pivot(index='household_id',columns='item',values='affinity')\nplt.figure(figsize=(10,4)); plt.imshow(matrix,aspect='auto',vmin=0,vmax=1); plt.colorbar(label='Affinity'); plt.xlabel('Context-slot item');"]),
('03_collaborative_filtering','Collaborative filtering','The saved experiment excludes cold-start households and uses validation-selected hyperparameters. These cells inspect the fitted models without retuning on test.',[
"import joblib\nartifact = joblib.load(ROOT/'results/models/ImplicitALS_11.joblib')\nmodel = artifact['model']\nprint(artifact['params'])",
"from src.recommend import recommend\nrecommend(ROOT/'results/models/ImplicitALS_11.joblib','H0000',daytype=0,k=5)",
"pd.read_csv(ROOT/'results/tables/metrics.csv').query('seed == 11')"]),
('04_hybrid_recommendation','Hybrid recommendation','The assumed task is added to the observed baseline. Savings are scenario outcomes. Allowed slots are enforced before ranking.',[
"from src.energy import simulated_tariff, hybrid_scores, simulate_day\nimport joblib\nmodel = joblib.load(ROOT/'results/models/ImplicitALS_11.joblib')['model']\npref = model.predict('H0000',np.arange(48))\ntrain = pd.read_csv(ROOT/'data/processed/train.csv')\npeak = train.groupby('slot').kwh.mean().reindex(range(48)).to_numpy()\nscore, parts = hybrid_scores(pref,simulated_tariff(),peak,36,8,[.4,.3,.2,.1])\nfeasible = np.arange(28,45)\nbest = int(feasible[np.argmax(score[feasible])])\nprint('Illustrative candidate:',f'{best//2:02d}:{30*(best%2):02d}')",
"recommendations = pd.read_csv(ROOT/'results/tables/recommendations.csv')\nrecommendations.query(\"household_id == 'H0000' and seed == 11 and daytype == 0\")",
"pd.read_csv(ROOT/'results/tables/energy.csv').groupby('strategy')[['savings','savings_percent','inconvenience']].mean()"]),
('05_results_visualisation','Results and uncertainty','Bootstrap units are households after averaging repeated seeds/dates. Synthetic performance is not evidence of real-world efficacy.',[
"metrics = pd.read_csv(ROOT/'results/tables/across_seed_summary.csv')\nmetrics.set_index('model').ndcg_mean.plot.barh(xlim=(0,1),xlabel='NDCG@5');",
"pd.read_csv(ROOT/'results/tables/paired_comparisons.csv')",
"from IPython.display import Image, display\ndisplay(Image(filename=str(ROOT/'results/figures/10a_ablation.png')))"
])]
for name,title,intro,codes in books:
    cells=[{'cell_type':'markdown','metadata':{},'source':f'# {title}\n\n{intro}\n\nRun the complete pipeline first. Select the project virtual environment as the notebook kernel.'}]
    for code in [setup]+codes: cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':code})
    for index,cell in enumerate(cells): cell['id']=f'cell-{index}'
    book={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}},'nbformat':4,'nbformat_minor':5}
    (root/'notebooks'/f'{name}.ipynb').write_text(json.dumps(book,indent=2))
if __name__=='__main__': print('Wrote five notebook tutorials.')
