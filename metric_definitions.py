"""Executable definitions from the supplied SciSciNet2 manual.

These small functions document/test definitions. Production analyses retain
the supplied paper outcomes; the history benchmark rebuilds only source
profiles and their member-to-paper distances.
"""
from collections import Counter
import numpy as np


def reference_source_pivot(current, history):
    x=np.asarray(current,dtype=float);h=np.asarray(history,dtype=float)
    if x.shape!=h.shape or x.ndim!=1:raise ValueError("Aligned source vectors required")
    if np.any(x<0) or np.any(h<0):raise ValueError("Source counts must be nonnegative")
    if not np.isfinite(x).all() or not np.isfinite(h).all():return np.nan
    den=np.linalg.norm(x)*np.linalg.norm(h)
    return np.nan if den==0 else float(np.clip(1-np.dot(x,h)/den,0,1))


def source_pair_new_share(reference_sources, previously_seen_pairs):
    """Unordered *reference-slot* pairs, including diagonal source pairs.

    previously_seen_pairs must use pairs seen in strictly earlier publication
    years. Same-year papers enter that history only after all are evaluated.
    """
    counts=Counter(reference_sources);n=sum(counts.values())
    if n<2:return np.nan
    sources=sorted(counts);numerator=0
    seen={tuple(sorted(p)) for p in previously_seen_pairs}
    for i,s in enumerate(sources):
        for t in sources[i:]:
            weight=counts[s]*(counts[s]-1)//2 if s==t else counts[s]*counts[t]
            if (s,t) not in seen:numerator+=weight
    return numerator/(n*(n-1)/2)


def weighted_linear_quantile(values, integer_weights, q):
    """Manual eq. 13.8, without expanding the repeated pair-score vector."""
    values=np.asarray(values,float);w=np.asarray(integer_weights)
    good=np.isfinite(values)&(w>0)
    values,w=values[good],w[good]
    if not len(values):return np.nan
    if np.any(w!=np.floor(w)) or not 0<=q<=1:raise ValueError("Integer weights / q")
    order=np.argsort(values,kind="stable");values=values[order];w=w[order].astype(np.int64)
    cumulative=np.cumsum(w);h=(int(cumulative[-1])-1)*q;j=int(np.floor(h));a=h-j
    left=values[np.searchsorted(cumulative,j,side="right")]
    right=values[np.searchsorted(cumulative,min(j+1,cumulative[-1]-1),side="right")]
    return float((1-a)*left+a*right)


def emb_field_history_distance(focal, prior_same_field_vectors, min_count=20):
    """Cosine to the sum of valid, individually unit-normalized prior papers.

    Caller supplies only same-primary-field papers in [y-L,y-1], default L=5
    in the manual. This is a backward history, not a future citation window.
    """
    u=np.asarray(focal,float);a=np.asarray(prior_same_field_vectors,float)
    if a.ndim!=2 or a.shape[1]!=u.size:raise ValueError("Embedding dimensions")
    un=np.linalg.norm(u)
    if not np.isfinite(u).all() or un==0:return np.nan
    norm=np.linalg.norm(a,axis=1);ok=np.isfinite(a).all(axis=1)&(norm>0)
    if int(ok.sum())<min_count:return np.nan
    center=(a[ok]/norm[ok,None]).sum(axis=0);cn=np.linalg.norm(center)
    return np.nan if cn==0 else float(np.clip(1-np.dot(u/un,center/cn),0,2))
