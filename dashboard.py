"""Local results explorer: streamlit run dashboard.py (run pipeline first)."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st
from src.energy import simulated_tariff
st.set_page_config(page_title='Energy recommendation experiment',layout='wide')
st.title('Residential energy recommendation experiment')
root=Path(__file__).resolve().parent
results=root/'results'; tables=results/'tables'
if not (tables/'metrics.csv').exists():
    st.info('Run python -m src.pipeline --config config.yaml from the project directory first.'); st.stop()
meta=json.loads((results/'metadata.json').read_text())
st.warning('All savings and flexible-task outcomes are simulated. Aggregate meter patterns do not establish appliance flexibility.')
st.caption(f"Data: {meta['data_source']}")
r=pd.read_csv(tables/'recommendations.csv')
user=st.selectbox('Household',sorted(r.household_id.unique()))
seed=st.selectbox('Model seed',sorted(r.seed.unique()))
context=st.radio('Recommendation context',['Weekday','Weekend'],horizontal=True)
selected=r[(r.household_id==user)&(r.seed==seed)&(r.daytype==(context=='Weekend'))].sort_values('rank')
st.subheader('Historical electricity consumption')
history=pd.concat([pd.read_csv(root/'data/processed'/f'{part}.csv') for part in ['train','validation','test']])
history=history[history.household_id==user].copy()
history['timestamp']=pd.to_datetime(history.timestamp,utc=True)
st.line_chart(history.set_index('timestamp')[['kwh']])
st.caption('kWh per half-hour; includes the saved historical partitions for inspection only.')
st.subheader('Illustrative tariff scenario')
st.line_chart(pd.DataFrame({'GBP/kWh':simulated_tariff()},index=[f'{s//2:02d}:{30*(s%2):02d}' for s in range(48)]))
st.caption(meta['tariff'])
st.subheader('Feasible hybrid recommendations')
st.dataframe(selected,use_container_width=True,hide_index=True)
if not selected.empty: st.info(selected.iloc[0].explanation)
st.subheader('Simulated outcomes for this household')
e=pd.read_csv(tables/'energy.csv'); e=e[(e.household_id==user)&(e.seed==seed)]
cols=[c for c in ['cost','baseline_cost','savings','savings_percent','shift_half_hours','inconvenience','constraint_violation'] if c in e]
st.dataframe(e.groupby('strategy')[cols].mean(),use_container_width=True)
st.bar_chart(e.groupby('strategy')[['cost','baseline_cost']].mean())
st.caption('Costs are GBP per complete augmented household-day. Time shifts are half-hour units; inconvenience is normalized by the allowed shift.')
st.subheader('Offline baseline comparison')
m=pd.read_csv(tables/'metrics.csv'); st.dataframe(m[m.seed==seed],use_container_width=True,hide_index=True)
st.caption('Behavioural proxy metrics; these do not measure recommendation acceptance or causal savings.')
with st.expander('Aggregate peaks and experiment metadata'):
    st.dataframe(pd.read_csv(tables/'aggregate_peaks.csv'),use_container_width=True)
    st.json(meta)
