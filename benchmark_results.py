"""Conditional history comparisons and compact publication figures.

The primary statistic is retained regardless of its result. Coordinates and
support are computed on the actual fixed estimation papers in each design.
"""
from pathlib import Path
import json, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
from matplotlib.lines import Line2D
from benchmark_profiles import active_config
from figures_pnas import load_settings, _style, check_layout, BLUE, ORANGE, TEAL, _compact_n

COLORS={'baseline':BLUE,'conditioned':ORANGE}
LABELS={'any_novel_pct':'New pair','novel_share_pct':'Pair share','z10_asinh':'z10',
        'z50_asinh':'z50','semantic_distance':'Semantic'}
KEYS=['scenario','design','outcome','spec']
GAP=['b_g','b_g2','b_mg']
COV=['v_gg','v_gg2','v_gmg','v_g2g2','v_g2mg','v_mgmg']


def mc_tails(observed,null):
    null=np.asarray(null,float)
    if not len(null) or not np.isfinite(null).all() or not np.isfinite(observed):
        raise ValueError('A complete finite reference distribution is required')
    B=len(null);left=(1+np.sum(null<=observed))/(B+1);right=(1+np.sum(null>=observed))/(B+1)
    median=float(np.median(null))
    return dict(observed=float(observed),null_median=median,
        null_q025=float(np.quantile(null,.025)),null_q975=float(np.quantile(null,.975)),
        excess=float(observed-median),p_left=float(left),p_right=float(right),
        p_two=float(min(1,2*min(left,right))),replicates=B,
        mcse_upper=float(np.sqrt(right*(1-right)/(B+1))))


def holm(pvalues,family_size=None):
    p=np.asarray(pvalues,float);result=np.full(len(p),np.nan);valid=np.flatnonzero(np.isfinite(p))
    order=valid[np.argsort(p[valid],kind='stable')];n=max(len(valid),family_size or len(valid))
    if len(order):result[order]=np.minimum(1,np.maximum.accumulate([(n-i)*p[j] for i,j in enumerate(order)]))
    return result


def _bool(x):return pd.Series(x).astype(str).str.lower().isin(['true','1']).to_numpy()


def plan_for(c):
    p=Path(c['_out'])/'scenario_plan.json'
    return json.loads(p.read_text()) if p.exists() else [dict(scenario=c['scenario'],role='primary',
            assignment_mode='permutation',run_common_measures=c.get('run_common_measures',True))]


def planned_families(plan):
    source=any(s.get('role')=='source' for s in plan)
    total=0
    for s in plan:
        if s.get('assignment_mode','permutation')!='permutation':continue
        n=1+5*bool(s.get('run_common_measures',True))
        if s.get('role')=='primary' and source:
            src=next(x for x in plan if x.get('role')=='source')
            n+=1+5*bool(src.get('run_common_measures',True))
        total+=n
    return dict(secondary_global=max(0,2*total-1),secondary_local=total)


def local_linear_vector(cells):
    if cells.empty:return None
    w=cells.weight.to_numpy(float);w=w/w.sum()
    m=cells.mean_pivot.to_numpy(float)-.5
    a=cells.g_from.to_numpy(float)-.2;b=cells.g_to.to_numpy(float)-.2
    return np.array([np.dot(w,b-a),np.dot(w,b*b-a*a),np.dot(w,m*(b-a))])


def local_estimates(stats,cells):
    L=local_linear_vector(cells)
    if L is None or not set(GAP+COV)<=set(stats):return pd.DataFrame()
    d=stats.copy()
    if not np.isfinite(d[GAP+COV].to_numpy(float)).all():return pd.DataFrame()
    d['estimate']=d[GAP].to_numpy(float)@L
    variance=(L[0]**2*d.v_gg+L[1]**2*d.v_g2g2+L[2]**2*d.v_mgmg+
        2*L[0]*L[1]*d.v_gg2+2*L[0]*L[2]*d.v_gmg+2*L[1]*L[2]*d.v_g2mg)
    if (variance< -1e-10).any():raise ValueError('Negative local contrast variance')
    d['se']=np.sqrt(np.maximum(variance,0));d['lower']=d.estimate-1.96*d.se;d['upper']=d.estimate+1.96*d.se
    return d


def model_availability(c):
    path=Path(c['_out'])/'model_availability.csv'
    if not path.exists():return pd.DataFrame()
    d=pd.read_csv(path)
    if not {'scenario','design','status'}<=set(d):raise ValueError('Invalid model-availability table')
    return d


def summarize(c,scenario,role='sensitivity'):
    out=Path(c['_out'])/'null'/scenario
    if (out/'SCENARIO_EMPTY').exists() or (out/'BENCHMARK_NOT_ESTIMABLE').exists():return None
    availability=model_availability(c)
    if len(availability):
        main=availability[availability.scenario.eq(scenario)&availability.design.eq('main')]
        if len(main)==1 and main.status.iloc[0] in ['pending','needs_correct_sample','empty'] and not (out/'BENCHMARK_MODELS_READY').exists():
            return None
    if not (out/'BENCHMARK_MODELS_READY').exists():raise FileNotFoundError(f'Complete notebook 05 or assemble completed model worlds: {scenario}')
    settings=json.loads((out/'scenario_settings.json').read_text());B=int(settings['replicates'])
    mode=settings.get('assignment_mode','cycle' if settings.get('forbid_self_coauthors') else 'permutation')
    tables=out/'model_tables';stats=pd.read_csv(tables/'all_stats.csv');profiles=pd.read_csv(tables/'all_profiles.csv')
    if not len(stats):raise ValueError('No completed models')
    base=dict(scenario=scenario,role=role,formal_reference=mode=='permutation',assignment_mode=mode)
    globals_=[];locals_=[];local_worlds=[]
    for key,d in stats.groupby(['design','outcome','spec'],sort=False):
        if sorted(d.world.tolist())!=list(range(B+1)):raise ValueError(f'Incomplete worlds: {scenario}, {key}')
        if d.n.nunique()!=1:raise ValueError('The actual estimation sample changed across worlds')
        obs=d[d.world.eq(0)].iloc[0]
        row={**base,**dict(zip(['design','outcome','spec'],key)),**mc_tails(obs.partial_r2_gap,d.loc[d.world.gt(0),'partial_r2_gap'])}
        row.update(n=int(obs.n),primary=role=='primary' and key==('main','any_novel_pct','conditioned'))
        globals_.append(row)
        path=tables/(key[0]+'_shared_support_cells.csv');cells=pd.read_csv(path) if path.exists() else pd.DataFrame()
        q=local_estimates(d,cells)
        local={**base,**dict(zip(['design','outcome','spec'],key)), 'n':int(obs.n),'edges':len(cells),
               'available':not q.empty,'replicates':B}
        if len(q):
            observed=q[q.world.eq(0)].iloc[0]
            local.update(mc_tails(observed.estimate,q.loc[q.world.gt(0),'estimate']),
                observed_lower=float(observed.lower),observed_upper=float(observed.upper))
            local_worlds.append(q)
        else:
            local['reason']='no all-world common domain' if cells.empty else 'coefficient covariance unavailable in existing cache'
            if len(availability) and 'local_support_status' in availability:
                state=availability[availability.scenario.eq(scenario)&availability.design.eq(key[0])]
                if len(state)==1 and str(state.local_support_status.iloc[0]).startswith('unavailable'):
                    local['reason']='support checkpoints unavailable'
        locals_.append(local)
    profile_rows=[]
    for key,d in profiles.groupby(['design','outcome','spec','mean_pivot'],sort=False):
        if sorted(d.world.tolist())!=list(range(B+1)):raise ValueError('Incomplete profile worlds')
        obs=d[d.world.eq(0)].iloc[0];null=d[d.world.gt(0)]
        supported=bool(_bool([obs.supported])[0]);null_ok=_bool(null.supported)
        row={**base,**dict(zip(['design','outcome','spec','mean_pivot'],key)),
            'n':int(obs.n),'observed_supported':supported,'null_supported_fraction':float(null_ok.mean()),
            'all_worlds_supported':supported and bool(null_ok.all()),'replicates':B,
            'observed_lower':float(obs.lower),'observed_upper':float(obs.upper),
            'observed_failure':str(obs.get('reason','')) if not supported else ''}
        if row['all_worlds_supported']:row.update(mc_tails(obs.estimate,null.estimate))
        else:
            row['observed']=float(obs.estimate)
            for v in ['null_median','null_q025','null_q975','excess','p_left','p_right','p_two']:row[v]=np.nan
        profile_rows.append(row)
    if local_worlds:pd.concat(local_worlds,ignore_index=True).to_csv(tables/'local_contrasts_by_world.csv',index=False)
    return out,pd.DataFrame(globals_),pd.DataFrame(profile_rows),pd.DataFrame(locals_)


def calibrate(results,plan):
    family=planned_families(plan)
    global_table=pd.concat([r[1] for r in results],ignore_index=True)
    local_table=pd.concat([r[3] for r in results],ignore_index=True)
    profiles=pd.concat([r[2] for r in results],ignore_index=True)
    secondary=_bool(global_table.formal_reference)&~_bool(global_table.primary)
    global_table['p_Holm_secondary']=np.nan
    global_table.loc[secondary,'p_Holm_secondary']=holm(global_table.loc[secondary,'p_right'],family['secondary_global'])
    global_table['family_size']=np.where(secondary,family['secondary_global'],1)
    global_table['minimum_attainable_adjusted_rank']=np.minimum(1,global_table.family_size/(global_table.replicates+1))
    global_table['resolution_allows_005']=global_table.minimum_attainable_adjusted_rank<=.05
    local_table['p_Holm_local']=np.nan
    valid=_bool(local_table.formal_reference)&local_table.spec.eq('conditioned')&_bool(local_table.available)
    if 'p_two' in local_table:
        local_table.loc[valid,'p_Holm_local']=holm(local_table.loc[valid,'p_two'],family['secondary_local'])
    local_table['family_size']=family['secondary_local']
    local_table['minimum_attainable_adjusted_rank']=np.minimum(1,2*family['secondary_local']/(local_table.replicates+1))
    primary=global_table[_bool(global_table.primary)&_bool(global_table.formal_reference)]
    gate=len(primary)==1 and primary.p_right.iloc[0]<=.05
    profiles['primary_gate_passed']=bool(gate)
    profiles['p_Holm_primary_profiles']=np.nan
    p=profiles.role.eq('primary')&profiles.design.eq('main')&profiles.spec.eq('conditioned')
    profiles.loc[p,'p_Holm_primary_profiles']=holm(profiles.loc[p,'p_left'],3)
    for out,_,_,_ in results:
        s=out.name;g=global_table[global_table.scenario.eq(s)];l=local_table[local_table.scenario.eq(s)];pr=profiles[profiles.scenario.eq(s)]
        for d,name in [(g,'global'),(l,'local'),(pr,'profile')]:d.to_csv(out/'model_tables'/f'null_{name}_comparisons.csv',index=False)
        row=g[g.design.eq('main')&g.outcome.eq('any_novel_pct')&g.spec.eq('conditioned')]
        if len(row)!=1:raise ValueError('One main formation comparison per scenario is required')
        row=row.iloc[0];focal=l[l.design.eq('main')&l.outcome.eq('any_novel_pct')&l.spec.eq('conditioned')]
        is_primary=bool(row.primary);formal=bool(row.formal_reference)
        decision=row.p_right if is_primary else row.p_Holm_secondary
        status=('excess_gap_association' if pd.notna(decision) and decision<=.05 else 'not_distinguished_from_reference_distribution') if formal else 'structural_reference_only'
        summary=dict(status=status,scenario=s,role=row.role,n=int(row.n),replicates=int(row.replicates),
            observed=float(row.observed),null_median=float(row.null_median),null_q025=float(row.null_q025),null_q975=float(row.null_q975),
            p_upper=float(row.p_right),excess=float(row.excess),primary=is_primary,formal_reference=formal,
            p_Holm_secondary=float(row.p_Holm_secondary) if pd.notna(row.p_Holm_secondary) else None,
            supported_primary_profiles=int(pr.loc[pr.design.eq('main')&pr.spec.eq('conditioned'),'all_worlds_supported'].sum()),
            common_local_edges=int(focal.edges.iloc[0]) if len(focal) else 0,
            secondary_global_family=family['secondary_global'],secondary_local_family=family['secondary_local'],
            tail_interpretation=('conditional reference under within-stratum exchangeability' if formal else
                                'uniform admissible single-cycle structural benchmark; ranks are not exact permutation p values'))
        (out/'benchmark_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
        (out/'BENCHMARK_RESULTS_READY').write_text('ready\n')
    return global_table,local_table,profiles


def estimation_ids(out,design='main'):
    ids=pd.read_csv(out/'model_tables/estimation_paper_ids.csv',dtype={'paper_id':'string'})
    wanted=ids.loc[ids.design.eq(design),'paper_id']
    if wanted.duplicated().any() or wanted.empty:raise ValueError(f'Invalid actual sample: {design}')
    return wanted.reset_index(drop=True)


def geometry(out,design='main'):
    tables=out/'model_tables';wanted=estimation_ids(out,design);B=int(json.loads((out/'scenario_settings.json').read_text())['replicates'])
    edges=np.linspace(0,1,21);fine=np.linspace(0,1,101);counts=[];cm=[];cg=[];means=[]
    for world in range(B+1):
        path=out/'panel.parquet' if world==0 else out/'replicates'/f'replicate_{world:04}.parquet'
        d=pd.read_parquet(path,columns=['paper_id','mean_pivot','grounding_gap'])
        d['paper_id']=d.paper_id.astype('string')
        if d.paper_id.duplicated().any():raise ValueError('Repeated geometry paper')
        d=d.set_index('paper_id').reindex(wanted)
        if len(d)!=len(wanted) or not np.isfinite(d.to_numpy(float)).all():raise ValueError('Missing actual estimation geometry')
        if ((d.grounding_gap< -1e-8)|(d.grounding_gap>d.mean_pivot+1e-8)|(d.mean_pivot>1+1e-8)).any():raise ValueError('Infeasible M/G geometry')
        h=np.histogram2d(d.mean_pivot,d.grounding_gap,bins=[edges,edges])[0];counts.append(h)
        cm.append(np.r_[0,np.cumsum(np.histogram(d.mean_pivot,bins=fine)[0])/len(d)])
        cg.append(np.r_[0,np.cumsum(np.histogram(d.grounding_gap,bins=fine)[0])/len(d)])
        means.append(dict(world=world,n=len(d),mean_M=d.mean_pivot.mean(),mean_G=d.grounding_gap.mean()))
    counts=np.array(counts);cm=np.array(cm);cg=np.array(cg);means=pd.DataFrame(means)
    means['overlap_with_observed']=np.minimum(counts,counts[0]).sum(axis=(1,2))/len(wanted)
    means.to_csv(tables/(design+'_geometry_by_world.csv'),index=False)
    np.savez_compressed(tables/(design+'_geometry_cells.npz'),counts=counts,edges=edges,cdf_M=cm,cdf_G=cg,cdf_x=fine)
    return dict(counts=counts,edges=edges,cdf_M=cm,cdf_G=cg,cdf_x=fine,worlds=means,n=len(wanted))


def canvas(rows,cols,height_mm,settings):
    fig,axes=plt.subplots(rows,cols,figsize=(settings.width_mm/25.4,height_mm/25.4),layout='constrained',squeeze=False)
    fig.get_layout_engine().set(w_pad=1.5/72,h_pad=4/72,wspace=.035,hspace=.04)
    return fig,axes


def panel_title(ax,letter,title):ax.set_title(letter+'  '+textwrap.fill(title,22),loc='left',fontweight='bold',pad=4)


def save_figure(fig,path,settings):
    check_layout(fig,path.stem,settings.strict_layout)
    fig.savefig(path.with_suffix('.pdf'));fig.savefig(path.with_suffix('.png'),dpi=settings.png_dpi)
    if settings.export_svg:fig.savefig(path.with_suffix('.svg'))
    previews=path.parent/'previews';previews.mkdir(exist_ok=True)
    fig.savefig(previews/(path.stem+'.png'),dpi=settings.preview_dpi);plt.close(fig)


def reference_intervals(ax,q,labels,xlabel):
    for i,r in enumerate(q.itertuples()):
        ax.plot([r.null_q025,r.null_q975],[i,i],lw=3,color='.75')
        ax.scatter(r.null_median,i,color='.4',s=9)
        ax.scatter(r.observed,i,color=ORANGE,s=15,marker='D',zorder=3)
    ax.set(yticks=range(len(q)),yticklabels=labels,ylim=(len(q)-.5,-.5),xlabel=xlabel)
    ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,3),useMathText=True)


def figure_statistics(out,s):
    tables=out/'model_tables';stats=pd.read_csv(tables/'all_stats.csv')
    glob=pd.read_csv(tables/'null_global_comparisons.csv');profiles=pd.read_csv(tables/'null_profile_comparisons.csv')
    local=pd.read_csv(tables/'null_local_comparisons.csv');mainlocal=local[local.design.eq('main')&local.spec.eq('conditioned')]
    validlocal=mainlocal[_bool(mainlocal.available)]
    supported=profiles[profiles.design.eq('main')&_bool(profiles.all_worlds_supported)]
    second=not validlocal.empty or not supported.empty
    fig,axes=canvas(2 if second else 1,3,126 if second else 66,s)
    for j,spec in enumerate(['baseline','conditioned']):
        d=stats[stats.design.eq('main')&stats.outcome.eq('any_novel_pct')&stats.spec.eq(spec)]
        null=d.loc[d.world.gt(0),'partial_r2_gap'];obs=float(d.loc[d.world.eq(0),'partial_r2_gap'].iloc[0])
        ax=axes[0,j];ax.hist(null,bins=min(30,max(5,int(np.sqrt(len(null))))),color=COLORS[spec],alpha=.75)
        ax.axvline(obs,color='black',lw=1.1,label='Observed')
        panel_title(ax,chr(65+j),spec.capitalize()+' gap block')
        ax.set(xlabel=r'Gap-block partial $R^2$',ylabel='Random worlds' if j==0 else '')
        ax.text(.98,.93,f'N = {int(d.n.iloc[0]):,}',transform=ax.transAxes,ha='right',va='top',fontsize=6.2,bbox=dict(facecolor='white',edgecolor='none',pad=1,alpha=.9))
        ax.legend(frameon=False,fontsize=6,loc='upper left')
        ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,3),useMathText=True)
    ax=axes[0,2];q=glob[glob.design.eq('common')&glob.spec.eq('conditioned')].copy()
    if q.empty:q=glob[glob.design.eq('main')].copy();labels=q.spec.str.capitalize().tolist();title='Main-sample comparisons'
    else:
        q['_order']=q.outcome.map({y:i for i,y in enumerate(LABELS)});q=q.sort_values('_order')
        labels=[LABELS[y] for y in q.outcome];title='Common-outcome sample'
    panel_title(ax,'C',title);reference_intervals(ax,q,labels,r'Gap-block partial $R^2$')
    ax.text(.97,.02,'N = '+', '.join(f'{n:,}' for n in q.n.unique()),transform=ax.transAxes,ha='right',va='bottom',fontsize=6)
    if second:
        for j,spec in enumerate(['baseline','conditioned']):
            ax=axes[1,j]
            l=local[local.design.eq('main')&local.spec.eq(spec)&_bool(local.available)]
            if len(l):
                panel_title(ax,chr(68+j),spec.capitalize()+' local contrast')
                reference_intervals(ax,l,[LABELS[y] for y in l.outcome],r'Outcome change for $\Delta G=0.05$')
                ax.axvline(0,color='.65',lw=.6)
                for i,r in enumerate(l.itertuples()):ax.plot([r.observed_lower,r.observed_upper],[i+.12,i+.12],color=COLORS[spec],lw=1)
            else:
                p=supported[supported.spec.eq(spec)].sort_values('mean_pivot')
                panel_title(ax,chr(68+j),spec.capitalize()+' fixed profiles')
                if len(p):reference_intervals(ax,p,[f'M = {m:.1f}' for m in p.mean_pivot],'Formation contrast (pp)')
                else:ax.set_axis_off()
        ax=axes[1,2];panel_title(ax,'F','All-world common domain')
        cells=pd.read_csv(tables/'main_shared_support_cells.csv')
        if len(cells):
            ax.scatter(cells.mean_pivot,cells.g_from,s=15+300*cells.weight,color=TEAL,alpha=.75)
            ax.set(xlabel='Mean distance, M',ylabel='Lower gap, G',xlim=(0,1),ylim=(0,1))
            ax.set_box_aspect(1)
        else:
            p=profiles[profiles.design.eq('main')&profiles.spec.eq('conditioned')]
            ax.bar(p.mean_pivot,p.null_supported_fraction,width=.09,color=ORANGE)
            ax.set(xlabel='Mean distance, M',ylabel='Supported random worlds',ylim=(0,1.05),xticks=[.4,.6,.8])
    save_figure(fig,out/'figures/Fig_B1_conditional_history_benchmark',s)


def figure_geometry(out,d,s):
    counts=d['counts'];obs=counts[0];median=np.median(counts[1:],axis=0);edges=d['edges'];n=d['n']
    occupied=np.any(counts>0,axis=0).any(axis=0);ymax=edges[np.flatnonzero(occupied)[-1]+1] if occupied.any() else 1
    centers=(edges[1:]+edges[:-1])/2;impossible=centers[:,None]>centers[None,:]
    fig,axes=canvas(2,3,126,s);top=max(float(obs.max()),float(median.max()),1)
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('white')
    for j,(a,title) in enumerate([(obs,'Observed configuration'),(median,'Median random configuration')]):
        ax=axes[0,j];panel_title(ax,chr(65+j),title)
        im=ax.imshow(np.ma.masked_where(impossible,a.T),origin='lower',extent=[0,1,0,1],aspect='auto',
                     norm=colors.LogNorm(vmin=1,vmax=max(1.01,top)),cmap=cmap)
        ax.fill_between(edges,edges,1,color='#eeeeee',lw=0)
        ax.set(xlabel='Mean distance, M',ylabel='Gap, G' if j==0 else '',ylim=(0,ymax))
    cb=fig.colorbar(im,ax=axes[0,:2],location='bottom',shrink=.82,pad=.04);cb.set_label('Papers per cell (log scale)',fontsize=6)
    diff=(obs-median)/n;lim=max(float(np.max(np.abs(diff))),1e-8)
    ax=axes[0,2];panel_title(ax,'C','Excess cell mass')
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('white')
    im=ax.imshow(np.ma.masked_where(impossible,diff.T),origin='lower',extent=[0,1,0,1],aspect='auto',cmap=cmap,
        norm=colors.SymLogNorm(linthresh=max(1/n,lim*.002),vmin=-lim,vmax=lim))
    ax.fill_between(edges,edges,1,color='#eeeeee',lw=0)
    ax.set(xlabel='Mean distance, M',ylim=(0,ymax))
    cb=fig.colorbar(im,ax=ax,location='bottom',shrink=.9,pad=.04,ticks=[-lim,0,lim])
    cb.ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v,_:f'{100*v:g}%'))
    cb.ax.xaxis.set_minor_locator(ticker.NullLocator())
    cb.set_label('Observed − median random mass',fontsize=6)
    for j,key,title in [(0,'cdf_M','Mean-distance distribution'),(1,'cdf_G','Gap distribution')]:
        ax=axes[1,j];q=d[key];x=d['cdf_x'];lo,hi=np.quantile(q[1:],[.025,.975],axis=0)
        ax.fill_between(x,lo,hi,color='.8',label='Random 95% range')
        ax.plot(x,np.median(q[1:],axis=0),color='.45',lw=.8);ax.plot(x,q[0],color=BLUE,label='Observed')
        panel_title(ax,chr(68+j),title);ax.set(xlabel='M' if j==0 else 'G',ylabel='Cumulative mass',ylim=(0,1.02))
        if j==0:ax.legend(frameon=False,fontsize=5.8,loc='upper left')
    ax=axes[1,2];panel_title(ax,'F','Configuration overlap')
    v=d['worlds'].loc[lambda q:q.world.gt(0),'overlap_with_observed']
    ax.hist(v,bins=min(25,max(5,int(np.sqrt(len(v))))),color=TEAL,alpha=.8)
    ax.set(xlabel=r'$\sum_c\min(p_c^{obs},p_c^{rand})$',ylabel='Random worlds',xlim=(0,1))
    save_figure(fig,out/'figures/Fig_B2_configuration_geometry',s)


def figure_samples(out,s):
    cover=pd.read_csv(out/'matching_coverage.csv');strata=pd.read_csv(out/'donor_strata.csv')
    assignments=[pd.read_csv(p) for p in sorted(out.glob('assignment_*.csv'))]
    assignment=pd.concat(assignments,ignore_index=True);rates=assignment.groupby('replicate')[['member_slots','self_history_slots','own_team_history_slots']].sum()
    sizepath=out/'model_tables/sample_sizes.csv';sizes=pd.read_csv(sizepath) if sizepath.exists() else pd.DataFrame()
    yearpath=out/'model_tables/sample_sizes_by_year.csv';years=pd.read_csv(yearpath) if yearpath.exists() else pd.DataFrame()
    fig,axes=canvas(2,3,120,s)
    ax=axes[0,0];panel_title(ax,'A','Sample retained by stage')
    if len(sizes):
        q=sizes[sizes.design.eq('main')].iloc[0];values=[int(cover.initial_papers.sum()),int(q.matched),int(q.complete),int(q.estimated)]
        ax.bar(range(4),values,color=['.8','.65',BLUE,ORANGE]);ax.set(xticks=range(4),xticklabels=['Selected','Matched','Complete','FE sample'],ylabel='Papers')
        ax.tick_params(axis='x',labelrotation=35)
        for i,v in enumerate(values):ax.text(i,v,_compact_n(v),ha='center',va='bottom',fontsize=6)
        ax.set_ylim(0,max(values)*1.22)
    else:
        ax.plot(cover.year,cover.initial_papers,label='Selected',color='.6');ax.plot(cover.year,cover.matched_papers,label='Matched',color=BLUE)
        ax.set(xlabel='Year',ylabel='Papers');ax.legend(frameon=False,fontsize=6)
    ax=axes[0,1];panel_title(ax,'B','Annual estimation sample')
    ax.plot(cover.year,cover.matched_papers,label='Matched',color=BLUE)
    if len(years):
        q=years[years.design.eq('main')&years.stage.eq('estimated')];ax.plot(q.year,q.papers,label='FE sample',color=ORANGE)
    ax.set(xlabel='Publication year',ylabel='Papers');ax.legend(frameon=False,fontsize=6)
    ax=axes[0,2];panel_title(ax,'C','Donors per matching stratum')
    ax.hist(np.log2(strata.n_donors),bins=22,color=BLUE,alpha=.8)
    ax.set(xlabel=r'$\log_2$(donors per stratum)',ylabel='Strata')
    for j,key,title in [(0,'self_history_slots','Own-history retention'),(1,'own_team_history_slots','Team-history retention')]:
        ax=axes[1,j];panel_title(ax,chr(68+j),title)
        ax.plot(rates.index,100*rates[key]/rates.member_slots,color=COLORS['baseline' if j==0 else 'conditioned'])
        ax.set(xlabel='Random world',ylabel='% member slots');ax.set_ylim(bottom=0)
    ax=axes[1,2]
    if {'mean_history_cosine','history_similarity_pairs'}<=set(assignment):
        q=assignment.dropna(subset=['mean_history_cosine','history_similarity_pairs']).copy()
        q['weighted']=q.mean_history_cosine*q.history_similarity_pairs
        q=q.groupby('replicate')[['weighted','history_similarity_pairs']].sum()
        panel_title(ax,'F','History-source similarity');ax.plot(q.index,q.weighted/q.history_similarity_pairs,color=TEAL)
        ax.set(xlabel='Sampled random world',ylabel='Mean cosine similarity',ylim=(0,1.02))
    else:
        panel_title(ax,'F','History-length range');ax.scatter(strata.min_history_refs,strata.max_history_refs,s=4,alpha=.4,color=BLUE)
        ax.set(xscale='log',yscale='log',xlabel='Minimum history refs',ylabel='Maximum history refs')
    save_figure(fig,out/'figures/Fig_B3_matching_and_retention',s)


def paired_figure(c,plan,geometries,s):
    broad=next((x for x in plan if x.get('role')=='primary'),None);source=next((x for x in plan if x.get('role')=='source'),None)
    if not broad or not source or source['scenario'] not in geometries:return
    a=Path(c['_out'])/'null'/broad['scenario'];b=Path(c['_out'])/'null'/source['scenario']
    ids_path=a/'model_tables/estimation_paper_ids.csv'
    ids=pd.read_csv(ids_path,dtype={'paper_id':'string'})
    if not ids.design.eq('bridge_main').any():return
    if estimation_ids(a,'bridge_main').tolist()!=estimation_ids(b).tolist():raise ValueError('Paired matching scenarios must use the identical ordered papers')
    ga=geometry(a,'bridge_main');gb=geometries[source['scenario']]
    sa=pd.read_csv(a/'model_tables/all_stats.csv');sb=pd.read_csv(b/'model_tables/all_stats.csv')
    sa=sa[sa.design.eq('bridge_main')&sa.outcome.eq('any_novel_pct')&sa.spec.eq('conditioned')].sort_values('world')
    sb=sb[sb.design.eq('main')&sb.outcome.eq('any_novel_pct')&sb.spec.eq('conditioned')].sort_values('world')
    if not np.isclose(sa.loc[sa.world.eq(0),'partial_r2_gap'].iloc[0],sb.loc[sb.world.eq(0),'partial_r2_gap'].iloc[0],atol=1e-9,rtol=1e-7):
        raise ValueError('Observed paired models differ despite identical papers')
    fig,axes=canvas(1,3,67,s);rows=[]
    fig.suptitle(f'Identical estimation papers, N = {ga["n"]:,}',fontsize=s.font_pt)
    ax=axes[0,0];panel_title(ax,'A','Same-paper gap association')
    for i,(name,d,g) in enumerate([('Broad',sa,ga),('Prior-source',sb,gb)]):
        r=mc_tails(d.loc[d.world.eq(0),'partial_r2_gap'].iloc[0],d.loc[d.world.gt(0),'partial_r2_gap'])
        ax.plot([r['null_q025'],r['null_q975']],[i,i],lw=4,color='.7');ax.scatter(r['null_median'],i,s=10,color='.4')
        ax.scatter(r['observed'],i,color=ORANGE,marker='D',s=16)
        rows.append(dict(matching=name,n=g['n'],**r,median_overlap=g['worlds'].overlap_with_observed.iloc[1:].median()))
    ax.set(yticks=[0,1],yticklabels=['Broad','Prior-source'],ylim=(-.6,1.6),xlabel=r'Gap-block partial $R^2$')
    ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,3),useMathText=True)
    ax=axes[0,1];panel_title(ax,'B','Same-paper coordinate overlap')
    for g,col,label in [(ga,BLUE,'Broad'),(gb,TEAL,'Prior-source')]:
        v=np.sort(g['worlds'].overlap_with_observed.iloc[1:]);ax.plot(v,np.arange(1,len(v)+1)/len(v),color=col,label=label)
    ax.set(xlabel='Overlap coefficient',ylabel='Cumulative random worlds',xlim=(0,1),ylim=(0,1.02));ax.legend(frameon=False,fontsize=6)
    ax=axes[0,2];panel_title(ax,'C','Same-paper mean distance')
    for g,col,label in [(ga,BLUE,'Broad'),(gb,TEAL,'Prior-source')]:
        ax.hist(g['worlds'].mean_M.iloc[1:],bins=20,histtype='step',color=col,label=label,lw=1.1)
    ax.axvline(ga['worlds'].mean_M.iloc[0],color='black',lw=1,label='Observed')
    ax.set(xlabel='Mean M across papers',ylabel='Random worlds');ax.legend(frameon=False,fontsize=5.8)
    root=Path(c['_out']);directory=root/'comparison_figures';directory.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(root/'same_sample_matching_comparison.csv',index=False)
    save_figure(fig,directory/'Fig_B4_same_sample_matching',s)


def sampling_comparison(c,plan,s):
    """Retain the paper-hash benchmark as a sampling sensitivity if available."""
    _style(s)
    if c.get('legacy_extension'):return
    previous=Path(c.get('previous_run_dir','outputs/conditional_history_null_compact'))
    if not previous.is_absolute():previous=Path(c.get('_project',Path(__file__).parent))/previous
    old=previous/'null/primary'
    current=next((x for x in plan if x.get('role')=='primary'),None)
    if not current:return
    new=Path(c['_out'])/'null'/current['scenario']
    if old.resolve()==new.resolve() or not (old/'BENCHMARK_MODELS_READY').exists() or not (old/'model_tables/all_stats.csv').exists():return
    rows=[]
    for label,out in [('Paper hash',old),('Member clusters',new)]:
        stats=pd.read_csv(out/'model_tables/all_stats.csv')
        q=stats[stats.design.eq('main')&stats.outcome.eq('any_novel_pct')]
        for spec,d in q.groupby('spec'):
            worlds=sorted(d.world.tolist())
            if worlds!=list(range(len(worlds))) or d.n.nunique()!=1:raise ValueError('Incomplete sampling-sensitivity worlds')
            row=mc_tails(d.loc[d.world.eq(0),'partial_r2_gap'].iloc[0],d.loc[d.world.gt(0),'partial_r2_gap'])
            rows.append(dict(sampling=label,spec=spec,n=int(d.n.iloc[0]),**row))
    table=pd.DataFrame(rows);root=Path(c['_out']);table.to_csv(root/'sampling_sensitivity.csv',index=False)
    q=table[table.spec.eq('conditioned')]
    fig,axes=canvas(1,3,65,s)
    panel_title(axes[0,0],'A','Gap association')
    reference_intervals(axes[0,0],q,[f'{r.sampling}\nN = {r.n:,}' for r in q.itertuples()],r'Gap-block partial $R^2$')
    panel_title(axes[0,1],'B','Sample size')
    axes[0,1].bar(range(len(q)),q.n,color=[BLUE,ORANGE]);axes[0,1].set(xticks=range(len(q)),xticklabels=['Paper\nhash','Member\nclusters'], ylabel='Papers')
    for i,r in enumerate(q.itertuples()):axes[0,1].text(i,r.n,f'{r.n:,}',ha='center',va='bottom',fontsize=6)
    axes[0,1].set_ylim(0,q.n.max()*1.22)
    panel_title(axes[0,2],'C','Reference rank')
    axes[0,2].bar(range(len(q)),q.p_right,color=[BLUE,ORANGE]);axes[0,2].set(xticks=range(len(q)),xticklabels=['Paper\nhash','Member\nclusters'],ylabel='Upper-tail rank',ylim=(0,1.05))
    for i,r in enumerate(q.itertuples()):axes[0,2].text(i,r.p_right,f'{r.p_right:.4f}',ha='center',va='bottom',fontsize=6)
    directory=root/'comparison_figures';directory.mkdir(exist_ok=True)
    save_figure(fig,directory/'Fig_B5_sampling_sensitivity',s)


def figures(c=None,scenario=None):
    c=c or active_config();scenario=scenario or c['scenario'];out=Path(c['_out'])/'null'/scenario
    if not (out/'BENCHMARK_RESULTS_READY').exists():
        r=summarize(c,scenario,'primary');calibrate([r],[dict(scenario=scenario,role='primary',run_common_measures=c.get('run_common_measures',True))])
    s=load_settings();_style(s);(out/'figures').mkdir(exist_ok=True)
    figure_statistics(out,s);geo=geometry(out);figure_geometry(out,geo,s);figure_samples(out,s)
    return out


def run_all(c=None):
    c=c or active_config();plan=plan_for(c);results=[]
    for x in plan:
        r=summarize(c,x['scenario'],x.get('role','sensitivity'))
        if r is not None:results.append(r)
    if not results:raise ValueError('No completed scenarios to summarize')
    primary_names=[x['scenario'] for x in plan if x.get('role')=='primary']
    if primary_names and primary_names[0] not in [r[0].name for r in results]:
        raise ValueError('The primary scenario is incomplete; assemble its completed worlds before exporting results')
    availability=model_availability(c)
    if len(availability):
        pending=availability[~availability.status.isin(['available','empty'])]
        if len(pending):
            print('Using completed estimates; these planned comparisons remain pending:')
            print(pending[['scenario','design','status']].to_string(index=False))
    g,l,p=calibrate(results,plan);root=Path(c['_out'])
    g.to_csv(root/'all_scenario_global_comparisons.csv',index=False)
    l.to_csv(root/'all_scenario_local_comparisons.csv',index=False)
    s=load_settings();_style(s);geometries={}
    for out,_,_,_ in results:
        (out/'figures').mkdir(exist_ok=True);figure_statistics(out,s)
        geo=geometry(out);geometries[out.name]=geo;figure_geometry(out,geo,s);figure_samples(out,s)
    paired_figure(c,plan,geometries,s)
    sampling_comparison(c,plan,s)
    return [r[0] for r in results]


if __name__=='__main__':run_all()
