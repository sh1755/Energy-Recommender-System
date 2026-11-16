"""Regenerate editable architecture diagrams with Graphviz (system package)."""
from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parents[1]
DIAGRAMS = {
'01_research_methodology': '''raw [label="LCL CSVs or labelled\nsynthetic demonstration"];
clean [label="Validate & audit\nMissingness / duplicates / time"];
split [label="Chronological split\nTrain → validation → test"];
fit [label="Train-only behavioural scores\nHousehold × context-slot matrix"];
tune [label="Validation model selection\nThen refit on train + validation"];
rank [label="Offline test ranking\nObserved behaviour proxy"];
sim [label="Constrained flexible-task\ncounterfactual simulation"];
report [label="Metrics & uncertainty\nSeparate empirical / simulated claims"];
raw -> clean -> split -> fit -> tune; tune -> rank; tune -> sim;
rank -> report; sim -> report;''',
'02_system_architecture': '''data [label="Data layer\nCSV loader + validation"];
features [label="Behaviour layer\nProfiles + interaction builder"];
models [label="Model layer\nSurprise baselines + implicit ALS"];
policy [label="Decision layer\nTariffs + constraints + weighted ranking"];
explain [label="Explanation layer\nScores + assumptions + cost factors"];
eval [label="Evaluation layer\nRanking / flexible-task simulation"];
outputs [label="Outputs\nCSV / joblib / figures / dashboard"];
data -> features -> models -> policy; policy -> explain; models -> eval;
policy -> eval; explain -> outputs; eval -> outputs;''',
'03_preprocessing_pipeline': '''read [label="Stream CSV chunks\nOriginal or generic headers"];
schema [label="Validate schema & units\nkWh per half-hour"];
time [label="Parse timestamps\nExplicit timezone policy"];
quality [label="Audit invalid readings\nMissing intervals / duplicates"];
split [label="Global chronological boundaries\nNo random row split"];
train [label="Fit thresholds & scales\nTraining portion only"];
transform [label="Apply fixed transforms\nValidation / test"];
export [label="Export splits & audit\nNo fabricated imputation"];
read -> schema -> time -> quality -> split -> train -> transform -> export;''',
'05_collaborative_filtering': '''meter [label="Training meter readings"];
score [label="Bounded behavioural affinity\nFrequency + magnitude + consistency"];
matrix [label="Household × time-slot matrix\nConfidence + observed mask"];
explicit [label="Pseudo-rating objective\nUser / item KNN, SVD, NMF"];
implicit [label="Implicit confidence objective\nWeighted alternating least squares"];
context [label="Context-aware item splitting\nWeekday / weekend × half-hour"];
rank [label="Predicted compatibility\nNot inferred appliance flexibility"];
meter -> score -> matrix; matrix -> explicit; matrix -> implicit;
implicit -> context; explicit -> rank; context -> rank;''',
'06_hybrid_optimisation': '''task [label="Assumed flexible task\nEnergy + original start"];
constraints [label="Feasible candidate set\nAvailability + maximum shift"];
preference [label="CF compatibility\nTraining-derived prediction"];
price [label="Exogenous signals\nSimulated tariff + train peak profile"];
objective [label="Weighted objective\nPreference − cost − peak − inconvenience"];
choose [label="Rank feasible starts\nDeterministic tie-breaking"];
explain [label="Factor contributions\nEstimated task savings + caveats"];
sim [label="Evaluate augmented profiles\nBaseline + task vs shifted task"];
task -> constraints; constraints -> objective; preference -> objective;
price -> objective; objective -> choose; choose -> explain; choose -> sim;''',
'11_temporal_evaluation': '''split [label="Global time split"];
train [label="Training period\nThresholds, scales, model fitting"];
val [label="Validation period\nHyperparameter selection"];
refit [label="Train + validation\nRefit selected configurations"];
test [label="Untouched future period\nFinal ranking evaluation"];
cold [label="Held-out households\nNo personal training history"];
boot [label="Household-level bootstrap\nPaired comparisons, seed summaries"];
split -> train -> val -> refit -> test -> boot; split -> cold -> boot;'''
}
def main():
    out=ROOT/'docs/diagrams'; out.mkdir(parents=True,exist_ok=True)
    fig=ROOT/'results/figures'; fig.mkdir(parents=True,exist_ok=True)
    for name,body in DIAGRAMS.items():
        source='digraph G {\ngraph [rankdir=TB,bgcolor="white",pad="0.3",nodesep="0.35",ranksep="0.4"];\nnode [shape=box,style="rounded,filled",fillcolor="#eef4f8",color="#274c65",fontname="DejaVu Sans",fontsize=12,margin="0.18,0.12"];\nedge [color="#526c7e",arrowsize=0.7];\n'+body+'\n}\n'
        dot=out/(name+'.dot'); dot.write_text(source)
        for ext in ['png','svg','pdf']:
            subprocess.run(['dot',f'-T{ext}','-Gdpi=180',str(dot),'-o',str(fig/(name+'.'+ext))],check=True)
if __name__=='__main__': main()
