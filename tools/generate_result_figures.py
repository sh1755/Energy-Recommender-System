"""Publication-style figures generated only from executed CSV outputs.
Run from repository root after the pipeline. No fabricated benchmark results.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.labelsize':10,'savefig.dpi':300,'svg.fonttype':'none','pdf.fonttype':42})
BLUE='#245c83'; ORANGE='#c46e35'; GREEN='#398575'
def save(fig,name):
    fig.tight_layout()
    for ext in ['png','svg','pdf']: fig.savefig(ROOT/'results/figures'/f'{name}.{ext}',bbox_inches='tight',facecolor='white')
    plt.close(fig)
def main():
    tables=ROOT/'results/tables'; processed=ROOT/'data/processed'
    meta=json.loads((ROOT/'results/metadata.json').read_text()); label='Synthetic demonstration' if 'synthetic' in meta['data_source'] else 'Supplied data; simulated task outcomes'
    data=pd.concat([pd.read_csv(processed/f'{s}.csv') for s in ['train','validation','test']],ignore_index=True)
    data['timestamp']=pd.to_datetime(data.timestamp,utc=True); local=data.timestamp.dt.tz_convert('Europe/London')
    data['weekday']=local.dt.dayofweek; data['week']=local.dt.strftime('%G-W%V')
    data['tariff_period']=np.where(data.slot<14,'off_peak',np.where(data.slot.between(32,39),'peak','normal'))
    daily=data.groupby(['household_id','date']).kwh.agg(['sum','count']).reset_index().rename(columns={'sum':'observed_kwh','count':'observed_half_hours'})
    daily.to_csv(tables/'daily_consumption.csv',index=False)
    data.groupby(['household_id','week']).kwh.agg(['sum','count']).to_csv(tables/'weekly_consumption.csv')
    data.groupby(['household_id','tariff_period']).kwh.agg(['sum','mean','count']).to_csv(tables/'peak_offpeak_consumption.csv')
    data.groupby(['household_id','daytype','slot']).kwh.mean().to_csv(tables/'household_profiles.csv')
    a=pd.read_csv(processed/'interactions.csv'); matrix=a.pivot(index='household_id',columns='item',values='affinity')
    fig,ax=plt.subplots(figsize=(9,5)); im=ax.imshow(matrix,aspect='auto',vmin=0,vmax=1,cmap='cividis'); ax.axvline(47.5,color='white',lw=1)
    ax.set(xticks=[0,24,47,48,72,95],xticklabels=['Weekday\n00:00','12:00','23:30','Weekend\n00:00','12:00','23:30'],ylabel='Training household index',title=f'{label}: training interaction matrix')
    fig.colorbar(im,ax=ax,label='Bounded behavioural affinity'); save(fig,'04_interaction_matrix')
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for daytype,lab,col in [(0,'Weekday',BLUE),(1,'Weekend',ORANGE)]:
        p=data[data.daytype==daytype].groupby('slot').kwh.agg(['mean','std']); axes[0,0].plot(p.index/2,p['mean'],label=lab,color=col)
    axes[0,0].set(xlabel='Local hour',ylabel='Mean kWh / half-hour',title='Daily consumption pattern'); axes[0,0].legend(frameon=False)
    for user in sorted(data.household_id.unique())[:5]:
        p=data[data.household_id==user].groupby('slot').kwh.mean(); axes[0,1].plot(p.index/2,p,label=user,lw=1.2)
    axes[0,1].set(xlabel='Local hour',ylabel='Mean kWh / half-hour',title='Five illustrative households'); axes[0,1].legend(frameon=False,ncol=2,fontsize=8)
    axes[1,0].hist(data.kwh.dropna(),bins=50,color=BLUE,alpha=.85); axes[1,0].set(xlabel='kWh / half-hour',ylabel='Reading count',title='Observed consumption distribution')
    p=data.groupby('weekday').kwh.mean(); axes[1,1].bar(p.index,p,color=GREEN); axes[1,1].set(xticks=range(7),xticklabels=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],ylabel='Mean kWh / half-hour',title='Weekly behaviour')
    fig.suptitle(label,y=1.01); save(fig,'07_consumption_patterns')
    m=pd.read_csv(tables/'across_seed_summary.csv'); u=pd.read_csv(tables/'uncertainty.csv')
    fig,ax=plt.subplots(figsize=(9,5)); y=np.arange(len(m)); bounds=u[(u.type=='ranking')&(u.measure=='ndcg')].set_index('model').reindex(m.model)
    means=bounds['mean'].to_numpy(); ax.barh(y,means,color=BLUE)
    ax.errorbar(means,y,xerr=np.maximum(0,np.vstack([means-bounds.ci_low.to_numpy(),bounds.ci_high.to_numpy()-means])),fmt='none',ecolor='#303030',capsize=3)
    ax.set(yticks=y,yticklabels=m.model,xlabel='NDCG@5 (household mean; 95% bootstrap interval)',title=f'{label}: future behavioural recovery',xlim=(0,1)); save(fig,'08_model_comparison')
    en=pd.read_csv(tables/'energy.csv'); h=en.groupby(['strategy','household_id']).agg(cost=('cost','mean'),baseline_cost=('baseline_cost','mean')).reset_index(); summary=h.groupby('strategy')[['cost','baseline_cost']].mean()
    fig,axes=plt.subplots(1,2,figsize=(11,4.5)); x=np.arange(len(summary)); axes[0].bar(x,summary.cost,color=BLUE); axes[0].set(xticks=x,xticklabels=summary.index,ylabel='GBP / augmented household-day',title='Simulated total energy cost'); axes[0].tick_params(axis='x',rotation=25)
    peaks=pd.read_csv(tables/'aggregate_peaks.csv'); p=peaks[peaks.strategy.isin(summary.index)].groupby('strategy')[['peak_kw','baseline_peak_kw']].mean().reindex(summary.index)
    reduction=100*(p.baseline_peak_kw-p.peak_kw)/p.baseline_peak_kw
    axes[1].bar(x,reduction,color=[GREEN if z>=0 else ORANGE for z in reduction]); axes[1].axhline(0,color='#333',lw=.8); axes[1].set(xticks=x,xticklabels=summary.index,ylabel='Aggregate peak reduction (%)',title='Signed reduction; negative = rebound'); axes[1].tick_params(axis='x',rotation=25)
    fig.suptitle(f'{label} · assumed added 1 kWh task',y=1.02); save(fig,'09_simulated_cost_and_peak')
    ab=pd.read_csv(tables/'ablation.csv'); baseline=en[en.strategy=='hybrid']; ab=pd.concat([baseline,ab]); g=ab.groupby('strategy')[['cost','baseline_cost','shift_half_hours']].mean()
    fig,axes=plt.subplots(1,2,figsize=(11,4.5)); labels=[s.replace('ablation_without_','Without ').replace('hybrid','Full hybrid') for s in g.index]; x=np.arange(len(g))
    axes[0].bar(x,100*(g.baseline_cost-g.cost)/g.baseline_cost,color=BLUE); axes[0].set(xticks=x,xticklabels=labels,ylabel='Simulated total bill savings (%)',title='Weight ablation'); axes[0].tick_params(axis='x',rotation=25)
    axes[1].bar(x,g.shift_half_hours/2,color=ORANGE); axes[1].set(xticks=x,xticklabels=labels,ylabel='Mean absolute time shift (hours)',title='Inconvenience proxy'); axes[1].tick_params(axis='x',rotation=25)
    fig.suptitle(label,y=1.02); save(fig,'10a_ablation')
    sen=pd.read_csv(tables/'sensitivity.csv'); parts=sen.strategy.str.split('_',expand=True); sen['weight']=parts[1]; sen['level']=parts[2].astype(float)
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
    for ax,(weight,g) in zip(axes.ravel(),sen.groupby('weight',sort=False)):
        g=g.groupby('level')[['cost','baseline_cost','shift_half_hours']].mean(); ax.plot(g.index,100*(g.baseline_cost-g.cost)/g.baseline_cost,'o-',color=BLUE); ax.set(title=f'Vary {weight}; other weights fixed',xlabel='Weight value',ylabel='Simulated bill savings (%)'); ax.grid(axis='y',alpha=.2)
    fig.suptitle(label,y=1.01); save(fig,'10b_weight_sensitivity')
    print('Exported 7 result-based figures in PNG, SVG and PDF; descriptive CSV tables.')
if __name__=='__main__': main()
