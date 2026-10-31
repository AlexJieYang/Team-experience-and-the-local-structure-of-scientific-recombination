"""Fixed-reference history reassignment, appendable by world and year."""
from pathlib import Path
import gc,json,shutil
import numpy as np
import pandas as pd
import scipy.sparse as sp
from benchmark_profiles import (config,activate,write_frame,guard,cosine_slots,
    aggregate_configuration,ensure_history_descriptors,hash64,atomic_json)


def strata_for_pool(pool,width,min_pool,match_source=False,dominance_width=None):
    eligible=pool.eligible.fillna(False).to_numpy(bool)&pool.history_refs.gt(0).to_numpy()
    bins=np.full(len(pool),-1,np.int64)
    bins[eligible]=np.floor(np.log2(pool.loc[eligible,'history_refs'].to_numpy(float))/width).astype(np.int64)
    keys=pd.DataFrame({'field_id':pool.field_id.astype('string'),'history_bin':bins})
    if match_source:
        if 'dominant_source_code' not in pool:raise ValueError('History descriptors required')
        keys['prior_source']=pool.dominant_source_code
        if dominance_width is not None:
            keys['source_share_bin']=np.floor(pool.dominant_source_share/float(dominance_width)).astype(np.int16)
    codes=np.full(len(pool),-1,np.int64)
    codes[eligible]=pd.factorize(pd.MultiIndex.from_frame(keys.loc[eligible]),sort=True)[0]
    order=np.argsort(codes,kind='stable');valid=order[codes[order]>=0]
    groups={}
    if len(valid):
        cuts=np.flatnonzero(np.diff(codes[valid]))+1
        groups={int(codes[g[0]]):g for g in np.split(valid,cuts) if len(g)>=min_pool}
    return codes,groups,np.isin(codes,list(groups))


def forbidden_by_group(slots,codes,groups):
    """Encode coauthors within the same matching group, using bounded arrays."""
    local=np.full(len(codes),-1,np.int64);sizes=np.zeros(len(codes),np.int64)
    values={k:[] for k in groups}
    for k,g in groups.items():local[g]=np.arange(len(g));sizes[g]=len(g)
    for _,part in slots.groupby('paper_row',sort=False):
        rows=np.unique(part.history_row.to_numpy(np.int64));rows=rows[local[rows]>=0]
        if not len(rows):continue
        order=np.argsort(codes[rows],kind='stable');rows=rows[order]
        for g in np.split(rows,np.flatnonzero(np.diff(codes[rows]))+1):
            k=int(codes[g[0]]);a=local[g];n=int(sizes[g[0]])
            values[k].append(np.repeat(a,len(a))*n+np.tile(a,len(a)))
    result={}
    for k,g in groups.items():
        diagonal=np.arange(len(g),dtype=np.int64)*(len(g)+1)
        result[k]=np.unique(np.concatenate(values[k]+[diagonal]))
    return result


def candidate(n,rng,mode='permutation'):
    if mode=='permutation':return rng.permutation(n)
    if mode!='cycle':raise ValueError('Unknown assignment mode')
    order=rng.permutation(n);p=np.empty(n,np.int64);p[order]=np.roll(order,-1)
    return p


def draw_group(n,rng,forbidden=None,max_attempts=20000,mode='permutation'):
    rows=np.arange(n,dtype=np.int64)
    for _ in range(1 if forbidden is None else max_attempts):
        p=candidate(n,rng,mode)
        if forbidden is None or not np.isin(rows*n+p,forbidden).any():return p
    raise RuntimeError('Restricted reference could not be drawn; completed worlds are retained')


def make_mapping(n_rows,groups,seed,year,replicate,forbidden=None,max_attempts=20000,mode='permutation'):
    # The unrestricted stream is identical to the earlier 199-world code.
    result=np.arange(n_rows,dtype=np.int64)
    for k,g in sorted(groups.items()):
        rng=np.random.default_rng(np.random.SeedSequence([int(seed),int(year),int(replicate),int(k)]))
        p=draw_group(len(g),rng,None if forbidden is None else forbidden[k],max_attempts,mode)
        result[g]=g[p]
    return result


def update_settings(path,settings):
    path=Path(path)
    if path.exists():
        old=json.loads(path.read_text())
        if settings['replicates']<old['replicates']:raise ValueError('Use a new scenario to decrease the number of worlds')
        defaults={'match_source':False,'source_dominance_bin_width':None,'assignment_mode':'permutation'}
        immutable=['scenario','history_bin_width_log2','min_donor_pool','forbid_self_coauthors','seed',
                   'match_source','source_dominance_bin_width','assignment_mode']
        if settings.get('match_source'):immutable+=['base_min_pool','base_history_width']
        if settings.get('forbid_self_coauthors'):
            immutable+=['strict_pilot_draws','strict_pilot_min_accepts','rejection_attempts']
        changed=[k for k in immutable if old.get(k,defaults.get(k))!=settings.get(k,defaults.get(k))]
        if changed:raise ValueError('Incompatible matching settings: '+', '.join(changed)+'. Choose a new scenario name.')
    atomic_json(path,settings)


def prepare_scenario(c):
    root=Path(c['_out']);out=root/'null'/c['scenario'];out.mkdir(parents=True,exist_ok=True)
    if not (root/'PROFILES_READY').exists():raise FileNotFoundError('Prepare source profiles first')
    settings={k:c.get(k) for k in ['scenario','history_bin_width_log2','min_donor_pool','forbid_self_coauthors',
        'rejection_attempts','seed','replicates','run_common_measures','match_source','source_dominance_bin_width',
        'assignment_mode','strict_pilot_draws','strict_pilot_min_accepts','role','legacy_extension','base_min_pool','base_history_width']}
    settings['match_source']=bool(settings['match_source']);settings['assignment_mode']=settings['assignment_mode'] or 'permutation'
    update_settings(out/'scenario_settings.json',settings)
    if (out/'SCENARIO_READY').exists() or (out/'SCENARIO_EMPTY').exists():return out
    reports=[];panels=[];coverage=[]
    for directory in sorted((root/'cache/years').iterdir()):
        y=int(directory.name);pool=pd.read_parquet(directory/'pool.parquet')
        slots=pd.read_parquet(directory/'slots.parquet');papers=pd.read_parquet(directory/'papers.parquet')
        codes,groups,supported=strata_for_pool(pool,c['history_bin_width_log2'],c['min_donor_pool'],
            c.get('match_source',False),c.get('source_dominance_bin_width'))
        forbidden=forbidden_by_group(slots,codes,groups) if c['forbid_self_coauthors'] else None
        if forbidden is not None:
            failed=[]
            for k,g in groups.items():
                rng=np.random.default_rng(np.random.SeedSequence([c['seed'],y,k,92837]))
                rows=np.arange(len(g));accepted=0
                for _ in range(c['strict_pilot_draws']):
                    p=candidate(len(g),rng,c['assignment_mode'])
                    accepted+=not np.isin(rows*len(g)+p,forbidden[k]).any()
                if accepted<c['strict_pilot_min_accepts']:failed.append(k)
            for k in failed:groups.pop(k);forbidden.pop(k)
            supported=np.isin(codes,list(groups))
        if c.get('match_source'):
            # Keep source-matched papers inside the broad matching population.
            _,_,broad=strata_for_pool(pool,c.get('base_history_width',.5),c.get('base_min_pool',30))
            supported &= broad
        good=supported[slots.history_row.to_numpy(np.int64)]
        keep_paper=np.bincount(slots.paper_row,weights=good,minlength=len(papers))==papers.team_size.to_numpy()
        oldrows=np.flatnonzero(keep_paper);subset=papers.iloc[oldrows].copy().reset_index(drop=True)
        keep=keep_paper[slots.paper_row.to_numpy()];ss=slots.loc[keep].copy()
        remap=np.full(len(papers),-1,np.int64);remap[oldrows]=np.arange(len(oldrows))
        ss['scenario_paper_row']=remap[ss.paper_row.to_numpy()]
        used=set(codes[ss.history_row.to_numpy(np.int64)].tolist())
        groups={k:g for k,g in groups.items() if k in used}
        keys=sorted(groups)
        np.savez_compressed(out/f'strata_{y}.npz',codes=codes,included_groups=np.array(keys,np.int64),
            group_sizes=np.array([len(groups[k]) for k in keys],np.int64),
            group_rows=np.concatenate([groups[k] for k in keys]) if keys else np.empty(0,np.int64),original_paper_rows=oldrows)
        if forbidden is not None:np.savez_compressed(out/f'forbidden_{y}.npz',**{str(k):forbidden[k] for k in keys})
        write_frame(ss,out/f'slots_{y}.parquet');panels.append(subset)
        coverage.append(dict(year=y,initial_papers=len(papers),matched_papers=len(subset)))
        for k,g in groups.items():
            pp=pool.iloc[g];reports.append(dict(year=y,group=k,field_id=str(pp.field_id.iloc[0]),n_donors=len(g),
                min_history_refs=pp.history_refs.min(),max_history_refs=pp.history_refs.max(),
                focal_members=int(ss.history_row.isin(g).sum())))
        print(c['scenario'],y,'matched',len(subset),'of',len(papers),flush=True)
    pd.DataFrame(coverage).to_csv(out/'matching_coverage.csv',index=False)
    pd.DataFrame(reports,columns=['year','group','field_id','n_donors','min_history_refs','max_history_refs','focal_members']).to_csv(out/'donor_strata.csv',index=False)
    panel=pd.concat(panels,ignore_index=True) if panels else pd.DataFrame()
    if panel.empty:
        if c.get('role')=='primary':raise ValueError('No complete broad-matched teams')
        (out/'SCENARIO_EMPTY').write_text('No complete teams meet the matching rules\n');return out
    write_frame(panel.sort_values(['year','paper_id']).reset_index(drop=True),out/'panel.parquet')
    (out/'SCENARIO_READY').write_text('ready\n');return out


def run(c):
    out=prepare_scenario(c)
    if (out/'SCENARIO_EMPTY').exists():return out
    root=Path(c['_out']);output=out/'replicates';output.mkdir(exist_ok=True)
    partroot=out/'replicate_parts';partroot.mkdir(exist_ok=True)
    finalized={b for b in range(1,c['replicates']+1) if (output/f'replicate_{b:04}.parquet').exists() and (out/f'assignment_{b:04}.csv').exists()}
    todo=[b for b in range(1,c['replicates']+1) if b not in finalized]
    if todo and (out/'PERMUTATIONS_READY').exists():(out/'PERMUTATIONS_READY').unlink()
    for directory in sorted((root/'cache/years').iterdir()):
        if not todo:break
        y=int(directory.name);setting=np.load(out/f'strata_{y}.npz');rows=setting['original_paper_rows']
        if not len(rows):continue
        pieces=np.split(setting['group_rows'],np.cumsum(setting['group_sizes'])[:-1])
        groups=dict(zip(map(int,setting['included_groups']),pieces))
        pool=pd.read_parquet(directory/'pool.parquet');slots=pd.read_parquet(out/f'slots_{y}.parquet')
        papers=pd.read_parquet(directory/'papers.parquet').iloc[rows].reset_index(drop=True)
        forbidden=None
        if c['forbid_self_coauthors']:
            ff=np.load(out/f'forbidden_{y}.npz');forbidden={int(k):ff[k] for k in ff.files}
        X=sp.load_npz(directory/'current.npz');H=sp.load_npz(directory/'histories.npz')
        xn=np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel());hn=np.sqrt(np.asarray(H.multiply(H).sum(axis=1)).ravel())
        hr=slots.history_row.to_numpy(np.int64);pr=slots.paper_row.to_numpy(np.int64);original_keys=pr*len(pool)+hr
        queried=np.unique(hr)
        if len(queried)>c.get('history_similarity_max_pairs',3000):
            ix=np.argsort(hash64(queried,c['seed']+y),kind='stable')[:c['history_similarity_max_pairs']];queried=queried[ix]
        for b in todo:
            dd=partroot/f'{b:04}';dd.mkdir(exist_ok=True);part=dd/f'{y}.parquet';diag=dd/f'{y}.json'
            if part.exists() and diag.exists():
                try:
                    saved=json.loads(diag.read_text())
                    if saved.get('replicate')==b and saved.get('year')==y:continue
                except (ValueError,OSError):pass
            mapping=make_mapping(len(pool),groups,c['seed'],y,b,forbidden,c['rejection_attempts'],c.get('assignment_mode','permutation'))
            donors=mapping[hr];distances=cosine_slots(X,H,pr,donors,c['slot_batch_rows'],(xn,hn))
            M,G=aggregate_configuration(distances,slots.scenario_paper_row.to_numpy(),len(papers),papers.team_size)
            write_frame(pd.DataFrame(dict(paper_id=papers.paper_id,mean_pivot=M,grounding_gap=G)),part)
            ratio=pool.history_refs.to_numpy()[donors]/pool.history_refs.to_numpy()[hr]
            record=dict(replicate=b,year=y,member_slots=len(slots),self_history_slots=int(np.sum(donors==hr)),
                own_team_history_slots=int(np.isin(pr*len(pool)+donors,original_keys).sum()),
                min_donor_history_ratio=float(ratio.min()),max_donor_history_ratio=float(ratio.max()))
            if b<=c.get('history_similarity_draws',20) and len(queried):
                distance=cosine_slots(H,H,queried,mapping[queried],c['slot_batch_rows'],(hn,hn))
                record.update(mean_history_cosine=float(np.mean(1-distance)),history_similarity_pairs=len(queried))
            atomic_json(diag,record)
        del X,H,xn,hn;gc.collect();guard(c)
        print(c['scenario'],'randomized year',y,flush=True)
    for b in todo:
        dd=partroot/f'{b:04}'
        result=pd.concat([pd.read_parquet(p) for p in sorted(dd.glob('*.parquet'))],ignore_index=True)
        if result.paper_id.duplicated().any():raise ValueError('Repeated focal paper')
        write_frame(result,output/f'replicate_{b:04}.parquet')
        pd.DataFrame([json.loads(p.read_text()) for p in sorted(dd.glob('*.json'))]).to_csv(out/f'assignment_{b:04}.csv',index=False)
        print(c['scenario'],'world',b,'/',c['replicates'],flush=True)
    if not c.get('keep_replicate_parts',False):
        for b in range(1,c['replicates']+1):
            dd=partroot/f'{b:04}'
            if dd.exists() and (output/f'replicate_{b:04}.parquet').exists() and (out/f'assignment_{b:04}.csv').exists():shutil.rmtree(dd)
    (out/'PERMUTATIONS_READY').write_text('ready\n');return out


def run_scenarios(c=None):
    c=dict(c or config())
    if c.get('source_matching',True):ensure_history_descriptors(c)
    primary=dict(c);primary.update(match_source=False,assignment_mode='permutation',role='primary')
    plan=[primary]
    if c.get('source_matching',True):
        source=dict(c);source.update(scenario='source_matched',match_source=True,assignment_mode='permutation',role='source',
            min_donor_pool=c['source_min_donor_pool'],source_dominance_bin_width=c.get('source_dominance_bin_width'),
            base_min_pool=c['min_donor_pool'],base_history_width=c['history_bin_width_log2'])
        plan.append(source)
        if c.get('include_sensitivity_scenarios',True):
            strict=dict(source);strict.update(scenario='source_no_team',forbid_self_coauthors=True,assignment_mode='cycle',role='sensitivity',run_common_measures=False)
            plan.append(strict)
    if c.get('include_sensitivity_scenarios',True):
        narrow=dict(primary);narrow.update(scenario='narrow_history',history_bin_width_log2=.25,role='sensitivity',run_common_measures=False)
        plan.append(narrow)
    atomic_json(Path(c['_out'],'scenario_plan.json'),[{k:s.get(k) for k in ['scenario','role','run_common_measures','match_source','assignment_mode']} for s in plan])
    activate(c)
    return [run(s) for s in plan]


if __name__=='__main__':run_scenarios()
