"""Manuscript inserts generated from completed empirical estimates."""
from pathlib import Path
import json
import pandas as pd
from benchmark_profiles import active_config


def tex_escape(s):
    for a,b in [('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:s=s.replace(a,b)
    return s


def export(c=None,scenario=None):
    c=c or active_config();scenario=scenario or c['scenario'];root=Path(c['_out'])/'null'/scenario
    if not (root/'BENCHMARK_RESULTS_READY').exists():raise FileNotFoundError('Complete notebooks 04–06 before exporting results')
    s=json.loads((root/'benchmark_summary.json').read_text());tables=root/'model_tables'
    g=pd.read_csv(tables/'null_global_comparisons.csv');p=pd.read_csv(tables/'null_profile_comparisons.csv');local=pd.read_csv(tables/'null_local_comparisons.csv')
    output=root/'manuscript';output.mkdir(exist_ok=True)
    paragraph=(f"We compared observed configurations with {s['replicates']} history reassignments on the same "
        f"{s['n']:,} actual estimation papers. We held focal references, outcomes, team slots, covariates, "
        "and fixed-effect labels constant and reassigned complete prior histories within the specified strata. "
        f"The reference-conditioned gap block had partial R-squared {s['observed']:.6f}, compared with a "
        f"random-history median of {s['null_median']:.6f} and a central 95% reference range of "
        f"[{s['null_q025']:.6f}, {s['null_q975']:.6f}] (upper-tail Monte Carlo rank {s['p_upper']:.4f}). ")
    if s['formal_reference']:
        if not s['primary']:
            adjusted=s.get('p_Holm_secondary')
            paragraph+=f"The secondary-family Holm-adjusted upper-tail rank was {adjusted:.4f}. " if adjusted is not None else ''
        if s['status']=='excess_gap_association':
            paragraph+='The gap block exceeds this conditional reference distribution. We assess the direction and comparability of this association on the shared coordinate domain. '
        else:
            paragraph+='We did not distinguish the conditioned gap association from this reference distribution. '
    else:
        paragraph+='This restricted single-cycle comparison reports Monte Carlo ranks within its structural reference distribution. '
    domain=local[local.design.eq('main')&local.outcome.eq('any_novel_pct')&local.spec.eq('conditioned')]
    if len(domain) and str(domain.iloc[0].available).lower()=='true':
        r=domain.iloc[0]
        paragraph+=(f"On {int(r.edges)} adjacent cell comparisons shared by the observed configuration and every random world, "
            f"the weighted formation contrast for a 0.05 increase in G was {r.observed:.4f} percentage points "
            f"(cluster-based 95% interval [{r.observed_lower:.4f}, {r.observed_upper:.4f}]); "
            f"the two-sided reference rank was {r.p_two:.4f}")
        if pd.notna(r.p_Holm_local):paragraph+=f" (Holm-adjusted {r.p_Holm_local:.4f})"
        paragraph+='. '
    elif len(domain) and str(domain.iloc[0].get('reason',''))=='support checkpoints unavailable':
        paragraph+='Local common-support comparisons are unavailable because their support files are missing. '
    else:
        paragraph+='No local coordinate contrast satisfied the all-world common-support rule. We therefore make no same-coordinate anchoring comparison from this benchmark. '
    geom=tables/'main_geometry_by_world.csv'
    if geom.exists():
        d=pd.read_csv(geom);median=d.loc[d.world.gt(0),'overlap_with_observed'].median()
        paragraph+=f"The median observed–random coordinate overlap was {median:.3f} on the prespecified 20-by-20 grid. "
    paragraph+=f"All-world support was available for {s['supported_primary_profiles']} of the three fixed formation profiles."
    sampling=Path(c['_out'])/'sampling_sensitivity.csv'
    if s['primary'] and sampling.exists():
        previous=pd.read_csv(sampling)
        previous=previous[previous.sampling.eq('Paper hash')&previous.spec.eq('conditioned')]
        if len(previous):
            r=previous.iloc[0]
            paragraph+=(f" Under paper-hash sampling, the conditioned comparison used {int(r.n):,} estimation papers "
                f"and had observed partial R-squared {r.observed:.6f}, random median {r.null_median:.6f}, "
                f"and upper-tail rank {r.p_right:.4f} across {int(r.replicates)} random worlds.")
    coverage=Path(c['_out'])/'model_availability.csv'
    if s['primary'] and coverage.exists():
        coverage=pd.read_csv(coverage)
        pending=coverage[~coverage.status.isin(['available','empty'])]
        if len(pending):
            paragraph+=' These summaries include completed estimates only; comparisons marked pending in the accompanying tables have not been estimated.'
            pending.to_csv(output/'pending_comparisons.csv',index=False)
        elif (output/'pending_comparisons.csv').exists():
            (output/'pending_comparisons.csv').unlink()
    (output/'benchmark_results.txt').write_text(paragraph+'\n',encoding='utf-8')
    (output/'benchmark_results.tex').write_text(tex_escape(paragraph).replace('R-squared',r'$R^2$')+'\n',encoding='utf-8')
    rows=[]
    for r in g.itertuples():
        label=tex_escape(f'{r.design}: {r.outcome}: {r.spec}')
        adjusted=f'{r.p_Holm_secondary:.4f}' if pd.notna(r.p_Holm_secondary) else '--'
        rows.append(f'{label} & {int(r.n):,} & {r.observed:.6f} & {r.null_median:.6f} & '
                    f'[{r.null_q025:.6f}, {r.null_q975:.6f}] & {r.p_right:.4f} & {adjusted} '+r'\\')
    # Requires booktabs and longtable in the manuscript preamble.
    header=(r'\toprule Design / outcome / specification & $N$ & Observed & Random median & Random 95\% range & Upper rank & Holm \\'+'\n'+r'\midrule')
    table=(r'\begingroup\scriptsize\setlength{\tabcolsep}{2pt}'+'\n'+
        r'\begin{longtable}{@{}p{0.27\textwidth}r r r r r r@{}}'+'\n'+
        r'\caption{Conditional history reference: global gap-block partial $R^2$.}\label{tab:history-null}\\'+'\n'+
        header+r'\endfirsthead'+'\n'+header+r'\endhead'+'\n'+'\n'.join(rows)+'\n'+
        r'\bottomrule\end{longtable}\endgroup'+'\n')
    (output/'benchmark_global_table.tex').write_text(table,encoding='utf-8')
    g.to_csv(output/'benchmark_global_table.csv',index=False)
    p.to_csv(output/'benchmark_fixed_profile_table.csv',index=False)
    local.to_csv(output/'benchmark_local_contrast_table.csv',index=False)
    supported=p[p.all_worlds_supported.astype(str).str.lower().eq('true')]
    supported.to_csv(output/'supported_fixed_profiles.csv',index=False)
    return output


if __name__=='__main__':print(export())
