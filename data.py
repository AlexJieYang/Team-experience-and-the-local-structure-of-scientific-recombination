"""Pandas analysis; Arrow is used only for bounded Parquet I/O (no SQL).

All histories/novelty/network measures come from existing files. No citation
window is imposed. Output is descriptive/associational, not a causal estimate.
"""
from pathlib import Path
import gc
import json
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

PROJECT_DIR = Path(__file__).resolve().parent
CORE = ['paper_id', 'year', 'source_id', 'is_conference', 'primary_field_id',
        'n_references', 'c_total', 'c_total_norm', 'c_total_pct', 'team_size',
        'mean_prior_papers', 'mean_prior_h', 'n_countries', 'n_institutions',
        'is_international', 'mean_ref_age', 'mean_ref_prior_citations']
PIVOT = ['paper_id', 'author_id', 'year', 'reference_source_pivot',
         'new_reference_source_share', 'prior_window_reference_count',
         'current_reference_count', 'pivot_observed']
AUTHOR = ['author_id', 'year', 'prior_papers', 'prior_h', 'career_age']
MODULES = {
    'novelty': ('metrics/source_novelty.parquet', [
        'new_source_pair_share', 'source_pair_z10', 'source_pair_z50',
        'source_pair_z_coverage', 'n_reference_sources', 'reference_source_hhi']),
    'diversity': ('metrics/team_diversity.parquet', [
        'profile_coverage', 'team_field_shannon', 'team_within_shannon',
        'team_between_js', 'team_cosine_distance', 'new_pair_share',
        'fresh_member_share', 'newcomer_share', 'prior_h_gini', 'prior_output_log_sd', 'prior_state_coverage']),
    'network': ('metrics/collaboration_network.parquet', [
        'network_author_coverage', 'prior_team_edge_density',
        'prior_team_shared_papers_mean', 'mean_prior_coauthor_degree',
        'mean_prior_burt_constraint', 'network_local_metric_coverage',
        'network_history_calendar_complete']),
    'idr': ('metrics/interdisciplinarity.parquet', [
        'ref_field_coverage', 'ref_field_rao', 'ref_field_shannon',
        'ref_outside_primary_share', 'idr_enough_refs']),
    'semantic': ('metrics/embedding_field_history.parquet', [
        'emb_field_history_count', 'emb_field_history_distance']),
}


def load_config(path=None):
    c=json.loads(Path(path or PROJECT_DIR/'config.json').read_text(encoding='utf-8'))
    assert c['year_start']<=c['year_end']<=c['observation_end']==2024
    assert 0<c['paper_fraction']<=1
    assert 0<c['history_coverage_min']<=1
    assert 0<c['profile_common_mass_min']<=1 and c['profile_context_min_count']>=1
    assert 2<=c['min_team_size']<=c['max_team_size']<=100
    assert Path(c['run_name']).name==c['run_name'] and c['run_name'] not in ('.','..')
    return c


def check_input_period(c):
    """Reject known time-window inconsistencies without generating reports."""
    root=Path(c['data_root']);ends=[];windows=[]
    meta=root/'config.json'
    if meta.exists():
        settings=json.loads(meta.read_text(encoding='utf-8-sig'))
        if settings.get('END_YEAR') is not None:ends.append(int(settings['END_YEAR']))
        value=settings.get('recent_measures',{}).get('PIVOT_YEARS')
        if value is not None:windows.append(int(value))
    selection=root/'qa/selection.parquet'
    if selection.exists() and 'observation_end' in pq.ParquetFile(selection).schema_arrow.names:
        ends.extend(pd.read_parquet(selection,columns=['observation_end']).observation_end.dropna().astype(int).unique())
    if ends and max(ends)>c['observation_end']:
        raise ValueError('Precomputed metrics include years after 2024; use a compatible snapshot.')
    if windows and set(windows)!={3}:raise ValueError('Source-pivot histories must cover three prior years.')


def preflight(c):
    root=Path(c['data_root'])
    tables=[('core/papers.parquet',CORE),('core/authors_year.parquet',AUTHOR),
            ('metrics/author_paper_pivot.parquet',PIVOT),
            ('metrics/source_novelty.parquet',['paper_id','new_source_pair_share','n_reference_sources','reference_source_hhi'])]
    for rel,columns in tables:
        path=root/rel
        if not path.exists():raise FileNotFoundError(path)
        missing=set(columns)-set(pq.ParquetFile(path).schema_arrow.names)
        if missing:raise KeyError(f'{rel}: missing required columns {sorted(missing)}')


def output_root(c):
    return PROJECT_DIR / 'outputs' / c['run_name']


def memory_guard(c, label=''):
    rss = psutil.Process().memory_info().rss / 2**30
    avail = psutil.virtual_memory().available / 2**30
    if rss > c['max_process_gb'] or avail < c['min_available_gb']:
        raise MemoryError(f'{label}: RSS={rss:.1f} GiB, available={avail:.1f} GiB. '
                          'Lower paper_fraction/batch_rows; close other kernels.')
    return rss


def exact_uint(s, name):
    """Never round a floating-point ID, even if it looks integral."""
    if s.isna().any() or pd.api.types.is_float_dtype(s.dtype):
        raise ValueError(f'{name}: missing/floating IDs prohibited; fix upstream inputs.')
    if pd.api.types.is_integer_dtype(s.dtype):
        if (s < 0).any():
            raise ValueError(f'{name}: negative ID')
        return s.astype('uint64')
    t = s.astype('string')
    if not t.str.fullmatch(r'\d+').all():
        raise ValueError(f'{name}: expected exact decimal numeric IDs')
    return t.astype('uint64')


def sample_ids(ids, fraction, seed):
    """Stable splitmix64 hash; independent of input order, batching and pandas version."""
    x = np.asarray(ids, dtype=np.uint64).copy()
    if fraction == 1:
        return np.ones(len(x), dtype=bool)
    with np.errstate(over='ignore'):
        x += np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)).astype('float64') / 2**53 < fraction


def read_batches(path, columns, batch_rows):
    pf = pq.ParquetFile(path)
    missing = set(columns) - set(pf.schema_arrow.names)
    if missing:
        raise KeyError(f'{path}: missing columns {sorted(missing)}')
    for batch in pf.iter_batches(batch_size=batch_rows, columns=columns, use_threads=True):
        # Nullable Arrow integers must stay integer: default conversion of a
        # nullable int64 ID to float64 would irreversibly lose large ID bits.
        def integer_mapper(typ):
            if pa.types.is_integer(typ):
                prefix = 'UInt' if pa.types.is_unsigned_integer(typ) else 'Int'
                return getattr(pd, f'{prefix}{typ.bit_width}Dtype')()
            return None
        d = batch.to_pandas(types_mapper=integer_mapper)
        for key in ('paper_id', 'author_id'):
            if key in d:
                d[key] = exact_uint(d[key], key)
        yield d


class YearSink:
    """At most one writer per year; retains no data frames between writes."""
    def __init__(self, directory):
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=True)
        self.writers = {}

    def write(self, frame):
        for year, part in frame.groupby('year', sort=False, observed=True):
            year = int(year)
            table = pa.Table.from_pandas(part.reset_index(drop=True), preserve_index=False)
            if year not in self.writers:
                self.writers[year] = pq.ParquetWriter(
                    self.path/f'{year}.parquet', table.schema, compression='zstd')
            self.writers[year].write_table(table)

    def close(self):
        for writer in self.writers.values():
            writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def partition(c,out):
    root=Path(c['data_root'])
    # Uniform PAPER sampling. All author relations of sampled papers are retained.
    for name,rel,cols in [('core','core/papers.parquet',CORE),
                          ('pivot','metrics/author_paper_pivot.parquet',PIVOT),
                          ('authors','core/authors_year.parquet',AUTHOR)]:
        with YearSink(out/'cache'/name) as sink:
            for i,d in enumerate(read_batches(root/rel,cols,c['batch_rows'])):
                d=d.loc[d.year.between(c['year_start'],c['year_end'])]
                if name!='authors':
                    d=d.loc[sample_ids(d.paper_id,c['paper_fraction'],c['sample_seed'])]
                sink.write(d)
                if i%40==0:print(f'Reading {name}: batch {i}; RSS {memory_guard(c):.1f} GiB',flush=True)
        gc.collect()


def aggregate_team(core,piv,hist,c):
    if core.paper_id.duplicated().any() or hist.author_id.duplicated().any():
        raise ValueError('Duplicate paper / author-year identifiers')
    core=core.loc[core.team_size.between(c['min_team_size'],c['max_team_size']) &
                  core.n_references.ge(c['min_current_references'])].copy()
    piv=piv.loc[piv.paper_id.isin(core.paper_id)].copy()
    if piv.duplicated(['paper_id','author_id']).any():raise ValueError('Duplicate author-paper relation')
    piv=piv.merge(hist.drop(columns='year'),on='author_id',how='left',validate='many_to_one')
    p=pd.to_numeric(piv.reference_source_pivot,errors='raise')
    if not p.dropna().between(-1e-6,1+1e-6).all():raise ValueError('Pivot outside [0,1]')
    valid=(piv.pivot_observed.fillna(False).astype(bool) & p.notna() &
           piv.prior_window_reference_count.ge(c['min_prior_references']) &
           piv.current_reference_count.ge(c['min_current_references']) &
           piv.career_age.between(c['career_age_min'],c['career_age_max']))
    piv['valid_pivot']=p.clip(0,1).where(valid)
    piv['history_reference_count']=piv.prior_window_reference_count.where(valid)
    g=piv.groupby('paper_id',sort=False,observed=True)
    a=g.agg(n_listed=('author_id','size'),n_history=('valid_pivot','count'),
            mean_pivot=('valid_pivot','mean'),minimum_pivot=('valid_pivot','min'),
            pivot_sd=('valid_pivot','std'),median_pivot=('valid_pivot','median'),
            mean_history_refs=('history_reference_count','mean'),min_history_refs=('history_reference_count','min')).reset_index()
    # Quantile-based alternative reduces reliance on a single extreme member.
    q=g.valid_pivot.quantile(.25).rename('pivot_q25').reset_index()
    a=a.merge(q,on='paper_id',validate='one_to_one')
    a['history_coverage']=a.n_history/a.n_listed
    a['grounding_gap']=(a.mean_pivot-a.minimum_pivot).clip(lower=0)
    a['q25_gap']=(a.mean_pivot-a.pivot_q25).clip(lower=0)
    # Reference member is selected by PRIOR OUTPUT, not by current-paper fit.
    # This label does not imply actual leadership or contribution.
    reference=piv.loc[valid].sort_values(['prior_papers','author_id'],ascending=[False,True],kind='stable').drop_duplicates('paper_id')
    reference=reference[['paper_id','author_id','prior_papers','prior_h','career_age']].rename(columns={
        'author_id':'reference_member_id','prior_papers':'reference_prior_papers',
        'prior_h':'reference_prior_h','career_age':'reference_career_age'})
    reference['reference_member_id']=reference.reference_member_id.astype('UInt64')
    # Alternative stable reference uses highest prior h, not current-paper fit.
    href=piv.loc[valid].sort_values(['prior_h','prior_papers','author_id'],ascending=[False,False,True],kind='stable').drop_duplicates('paper_id')
    href=href[['paper_id','author_id']].rename(columns={'author_id':'h_reference_id'})
    href['h_reference_id']=href.h_reference_id.astype('UInt64')
    reference=reference.merge(href,on='paper_id',validate='one_to_one')
    a=a.merge(reference,on='paper_id',how='left',validate='one_to_one')
    a=a.merge(core,on='paper_id',how='inner',validate='one_to_one')
    if not a.n_listed.eq(a.team_size).all():raise ValueError('Incomplete author list: pivot/core versions differ')
    a=a.loc[a.n_history.ge(2)&a.history_coverage.ge(c['history_coverage_min'])].copy()
    if not a.grounding_gap.le(a.mean_pivot+1e-6).all():raise ValueError('Impossible mean-gap geometry')
    return a


def build_teams(c,out):
    rows=[]
    for year in range(c['year_start'],c['year_end']+1):
        paths=[out/'cache'/n/f'{year}.parquet' for n in ['core','pivot','authors']]
        if not all(p.exists() for p in paths):continue
        core,piv,hist=[pd.read_parquet(p) for p in paths]
        a=aggregate_team(core,piv,hist,c)
        rows.append(dict(year=year,sampled_candidate_papers=len(core),qualified_papers=len(a),
                         reference_members=a.reference_member_id.nunique()))
        if len(a):a.to_parquet(out/'cache/base'/f'{year}.parquet',index=False,compression='zstd')
        print(f'{year}: {len(a):,} qualified papers',flush=True)
        del core,piv,hist,a;gc.collect();memory_guard(c)
    pd.DataFrame(rows).to_csv(out/'tables/sample_composition.csv',index=False)


def join_existing_metrics(c,out):
    frames=[pd.read_parquet(p,columns=['paper_id','year']) for p in sorted((out/'cache/base').glob('*.parquet'))]
    if not frames:raise ValueError('No qualified papers; inspect sample settings')
    index=pd.concat(frames,ignore_index=True)
    if index.paper_id.duplicated().any():raise ValueError('Paper appears in multiple years')
    lookup=pd.Series(index.year.to_numpy(),index=pd.Index(index.paper_id))
    del frames,index
    for name,(rel,cols) in MODULES.items():
        path=Path(c['data_root'])/rel
        if not path.exists():continue
        available=pq.ParquetFile(path).schema_arrow.names
        cols=[k for k in cols if k in available]
        if not cols:continue
        with YearSink(out/'cache'/name) as sink:
            for i,d in enumerate(read_batches(path,['paper_id']+cols,c['batch_rows'])):
                yr=d.paper_id.map(lookup);keep=yr.notna()
                d=d.loc[keep].copy();d['year']=yr.loc[keep].astype('int16');sink.write(d)
                if i%80==0:memory_guard(c,name)
        print('Joined',name,flush=True)


def finish_tables(c,out):
    counts=[]
    for path in sorted((out/'cache/base').glob('*.parquet')):
        d=pd.read_parquet(path)
        for name,(_,cols) in MODULES.items():
            p=out/'cache'/name/path.name
            if p.exists():
                z=pd.read_parquet(p).drop(columns='year')
                if z.paper_id.duplicated().any():raise ValueError(f'Duplicate paper in {name}')
                d=d.merge(z,on='paper_id',how='left',validate='one_to_one')
            for col in cols:
                if col not in d:d[col]=np.nan
        d.loc[~d.idr_enough_refs.fillna(False).astype(bool),['ref_field_rao','ref_field_shannon']]=np.nan
        d.loc[d.profile_coverage.lt(c['history_coverage_min'])|d.profile_coverage.isna(),
              ['team_within_shannon','team_between_js']]=np.nan
        d.loc[~d.network_history_calendar_complete.fillna(False).astype(bool),
              ['prior_team_edge_density','mean_prior_coauthor_degree']]=np.nan
        d.loc[d.prior_state_coverage.lt(c['history_coverage_min'])|d.prior_state_coverage.isna(),
              ['prior_h_gini','prior_output_log_sd']]=np.nan
        for col in ['c_total','c_total_pct','new_source_pair_share']:
            x=pd.to_numeric(d[col],errors='raise')
            if col!='c_total' and not x.dropna().between(0,1).all():raise ValueError(f'Invalid {col}')
            if col=='c_total' and (x.dropna().lt(0).any() or x.dropna().gt(2**53).any()):raise ValueError('Invalid citation counts')
            d[col]=x.to_numpy(dtype=float,na_value=np.nan)
        d['citation_percentile']=100*d.c_total_pct
        d['novel_share_pct']=100*d.new_source_pair_share
        d['any_novel']=np.where(d.new_source_pair_share.notna(),(d.new_source_pair_share>0).astype(float),np.nan)
        d['high_recognition']=np.where(d.c_total_pct.notna(),(100*d.c_total_pct>=c['high_recognition_pct']).astype(float),np.nan)
        d['joint_success']=d.any_novel*d.high_recognition
        d['any_novel_pct']=100*d.any_novel;d['joint_success_pct']=100*d.joint_success
        d['high_recognition_pct']=100*d.high_recognition
        for cutoff in [75,95]:
            h=np.where(d.c_total_pct.notna(),(100*d.c_total_pct>=cutoff).astype(float),np.nan)
            d[f'high{cutoff}_pct']=100*h
            d[f'joint{cutoff}_pct']=100*h*d.any_novel
        valid_sources=d.n_reference_sources.dropna()
        if (valid_sources.lt(1)|valid_sources.mod(1).ne(0)).any():raise ValueError('Invalid reference source counts')
        if not d.reference_source_hhi.dropna().between(0,1).all():raise ValueError('Source HHI outside [0,1]')
        d['source_hhi_pct']=100*d.reference_source_hhi
        d['source_count_bin']=pd.cut(d.n_reference_sources,bins=[0,2,4,8,16,32,64,128,np.inf],labels=False).astype('Int16')
        d['hhi_bin']=np.floor(d.reference_source_hhi*10).clip(upper=9).astype('Int16')
        d['ref_count_bin']=pd.cut(d.n_references,bins=[0,10,15,20,30,40,60,100,200,np.inf],labels=False).astype('Int16')
        d['positive_novel_share']=d.novel_share_pct.where(d.any_novel.eq(1))
        d['log_citations']=np.log1p(d.c_total)
        d['semantic_distance']=d.emb_field_history_distance
        d['z10_asinh']=np.arcsinh(d.source_pair_z10);d['z50_asinh']=np.arcsinh(d.source_pair_z50)
        for dest,source in [('log_team','team_size'),('log_refs','n_references'),
            ('log_team_h','mean_prior_h'),('log_team_output','mean_prior_papers'),
            ('log_ref_output','reference_prior_papers'),('log_ref_h','reference_prior_h'),
            ('log_degree','mean_prior_coauthor_degree'),('log_min_history_refs','min_history_refs'),
            ('log_mean_history_refs','mean_history_refs'),('log_n_sources','n_reference_sources'),
            ('log_ref_citation_base','mean_ref_prior_citations')]:
            d[dest]=np.log1p(d[source])
        # Deterministic paper halves for the stability comparison.
        # previous results already informed the present design.
        d['split_half']=sample_ids(d.paper_id,.5,c['sample_seed']+103).astype('int8')
        for col in ['paper_id','reference_member_id','h_reference_id','source_id','primary_field_id']:
            if pd.api.types.is_float_dtype(d[col]) and col in ['source_id','primary_field_id']:
                raise ValueError('ID must not pass through floating point')
            d[col]=d[col].astype('string')
        d['is_conference']=d.is_conference.astype('Int8')
        for col in d.select_dtypes('float64').columns:
            if col!='c_total':d[col]=d[col].astype('float32')
        d.to_parquet(out/'data'/path.name,index=False,compression='zstd')
        counts.append(dict(year=int(path.stem),papers=len(d),citation_known=int(d.citation_percentile.notna().sum()),
          novelty_known=int(d.any_novel.notna().sum()),semantic_known=int(d.semantic_distance.notna().sum())))
        del d;gc.collect();memory_guard(c)
    pd.DataFrame(counts).to_csv(out/'tables/annual_sample.csv',index=False)


def run(c=None):
    c=c or load_config();preflight(c);check_input_period(c);out=output_root(c)
    if out.exists():raise FileExistsError(f'{out} exists; choose a new run_name')
    for rel in ['cache/base','data','tables','figures','models']:(out/rel).mkdir(parents=True,exist_ok=True)
    (out/'config_used.json').write_text(json.dumps(c,indent=2),encoding='utf-8')
    partition(c,out);build_teams(c,out);join_existing_metrics(c,out);finish_tables(c,out)
    (out/'DATA_READY').write_text('ready',encoding='utf-8')
    return out


if __name__=='__main__':run()
