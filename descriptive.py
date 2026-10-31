"""Bounded descriptive summaries; no figure rendering in the data notebook."""
from pathlib import Path
import numpy as np
import pandas as pd
from data import load_config,output_root,memory_guard

def classify(m,minimum):return 2*(np.asarray(m)>=.5).astype(int)+(np.asarray(minimum)>=.25).astype(int)

def describe(c=None):
 c=c or load_config();out=output_root(c)
 if not(out/'DATA_READY').exists():raise RuntimeError('Run notebook 01 first')
 cols=['mean_pivot','minimum_pivot','grounding_gap','pivot_sd','team_size','history_coverage',
       'citation_percentile','novel_share_pct','any_novel','high_recognition','joint_success','semantic_distance',
       'n_reference_sources','reference_source_hhi','source_hhi_pct','log_min_history_refs','log_mean_history_refs','prior_h_gini','prior_output_log_sd']
 moments=[];allcells=[];cohorts=[];flows=[];sample=[];rng=np.random.default_rng(c['sample_seed'])
 for path in sorted((out/'data').glob('*.parquet')):
  d=pd.read_parquet(path,columns=cols+['year'])
  for v in d:d[v]=pd.to_numeric(d[v],errors='raise').to_numpy(dtype=float,na_value=np.nan)
  d['regime']=classify(d.mean_pivot,d.minimum_pivot)
  for v in cols:
   z=d[v].dropna();moments.append(dict(variable=v,n=len(z),sum=z.sum(),sq=np.square(z).sum(),lo=z.min(),hi=z.max()))
  d['mbin']=np.minimum((d.mean_pivot*20).astype(int),19)
  d['gbin']=np.minimum((d.grounding_gap*20).astype(int),19)
  d['cohort']=1990+5*((d.year-1990)//5)
  for y in ['citation_percentile','any_novel','novel_share_pct','joint_success','n_reference_sources','reference_source_hhi']:
   z=d.groupby(['mbin','gbin'])[y].agg(['count','sum']).reset_index();z['outcome']=y;allcells.append(z)
   z=d.groupby(['cohort','regime'])[y].agg(['count','sum']).reset_index();z['outcome']=y;cohorts.append(z)
  z=d.dropna(subset=['any_novel','high_recognition']).copy()
  z['novelty_class']=np.where(z.any_novel.eq(1),'New source pair','No new source pair')
  z['recognition_class']=np.where(z.high_recognition.eq(1),'High recognition','Other citation ranks')
  flows.append(z.groupby(['regime','novelty_class','recognition_class']).size().rename('papers').reset_index())
  d['_priority']=rng.random(len(d));sample.append(d.nsmallest(c['plot_max_rows'],'_priority'))
  if sum(len(z) for z in sample)>2*c['plot_max_rows']:sample=[pd.concat(sample).nsmallest(c['plot_max_rows'],'_priority')]
  memory_guard(c)
 s=pd.concat(sample,ignore_index=True).nsmallest(c['plot_max_rows'],'_priority').drop(columns='_priority')
 s.to_parquet(out/'tables/plot_sample.parquet',index=False)
 m=pd.DataFrame(moments).groupby('variable').agg(n=('n','sum'),total=('sum','sum'),sq=('sq','sum'),minimum=('lo','min'),maximum=('hi','max'))
 m['mean']=m.total/m.n.replace(0,np.nan);m['sd']=np.sqrt(((m.sq-m.total**2/m.n.replace(0,np.nan))/(m.n-1).where(m.n>1)).clip(lower=0))
 m[['n','mean','sd','minimum','maximum']].to_csv(out/'tables/descriptive_statistics.csv')
 cells=pd.concat(allcells).groupby(['outcome','mbin','gbin'])[['count','sum']].sum().reset_index();cells['mean']=cells['sum']/cells['count'].replace(0,np.nan)
 cells.to_csv(out/'tables/observed_cells.csv',index=False)
 co=pd.concat(cohorts).groupby(['outcome','cohort','regime'])[['count','sum']].sum().reset_index();co['mean']=co['sum']/co['count'].replace(0,np.nan)
 co.to_csv(out/'tables/cohort_composition.csv',index=False)
 flow=pd.concat(flows).groupby(['regime','novelty_class','recognition_class']).papers.sum().reset_index()
 flow.to_csv(out/'tables/alluvial_composition.csv',index=False)
 return m[['n','mean','sd','minimum','maximum']]
