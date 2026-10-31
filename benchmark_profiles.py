"""Bounded pandas / Arrow I/O and sparse source-profile reconstruction.

No SQL. Raw reference edges are reduced to the research core, deduplicated,
and counted by the core source of the cited paper. Authors receive the full
source vector of each prior paper, as specified in manual section 13.1.
"""
from pathlib import Path
import gc, json, os, shutil
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

PROJECT_DIR=Path(__file__).resolve().parent
BASE_COLUMNS=[
 "paper_id","year","reference_member_id","source_id","primary_field_id","is_conference",
 "team_size","mean_pivot","grounding_gap","any_novel_pct","novel_share_pct",
 "z10_asinh","z50_asinh","semantic_distance","log_refs","log_team_h","log_team_output",
 "log_ref_output","log_ref_h","log_n_sources","reference_source_hhi","mean_ref_age",
 "log_ref_citation_base","n_reference_sources","source_count_bin","hhi_bin","ref_count_bin"]
PIVOT_COLUMNS=["paper_id","author_id","year","reference_source_pivot",
               "prior_window_reference_count","current_reference_count"]


def config(project=PROJECT_DIR):
    project=Path(project).resolve()
    base=json.loads((project/"config.json").read_text(encoding="utf-8-sig"))
    b=json.loads((project/"benchmark_config.json").read_text(encoding="utf-8-sig"))
    if not 1500<=b["year_start"]<=b["year_end"]<=2024:raise ValueError("Publication years")
    if b["history_years"]!=3:raise ValueError("Source-history reconstruction uses three prior years")
    if b["replicates"]<1 or b["min_donor_pool"]<2:raise ValueError("Permutation settings")
    if not 0<b["donor_fraction"]<=1:raise ValueError("donor_fraction")
    if b["history_bin_width_log2"]<=0:raise ValueError("Positive history-bin width")
    if b.get("focal_sampling","paper_hash")=="reference_member_clusters":
        if not 0<b["reference_member_fraction"]<=1:raise ValueError("reference_member_fraction")
    for value in [b["run_name"],b["scenario"]]:
        if Path(value).name!=value or value in (".",".."):raise ValueError("Simple run/scenario name")
    b["_project"]=str(project);b["_data_root"]=base["data_root"]
    b["_analysis_run"]=str(Path(b["analysis_run_dir"]) if b["analysis_run_dir"] else project/"outputs"/base["run_name"])
    b["_out"]=str(project/"outputs"/b["run_name"])
    return b


def atomic_json(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.partial')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(tmp,path)


def activate(c):
    """Keep notebooks 04, 05 and 06 on the same selected run."""
    path=Path(c["_project"])/"active_benchmark.json"
    atomic_json(path,c)
    return c


def active_config(project=PROJECT_DIR):
    path=Path(project)/"active_benchmark.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else config(project)


def existing_config(path,project=PROJECT_DIR):
    """Append worlds to an immutable completed history cache, without raw I/O."""
    c=config(project);path=Path(path).resolve()
    if not (path/"PROFILES_READY").exists():raise FileNotFoundError("Existing PROFILES_READY run required")
    saved=json.loads((path/"profile_settings.json").read_text())
    for k,v in saved.items():
        if k!="files":c[k]=v
    c.update(_project=str(Path(project).resolve()),_out=str(path),run_name=path.name,
             focal_sampling="paper_hash",legacy_extension=True,source_matching=False,
             include_sensitivity_scenarios=False)
    settings=path/"null/primary/scenario_settings.json"
    if not settings.exists():raise FileNotFoundError("Existing primary scenario settings required")
    original=json.loads(settings.read_text())
    for k,v in original.items():
        if k!="replicates":c[k]=v
    c["replicates"]=max(config(project)["replicates"],original["replicates"])
    return activate(c)


def guard(c):
    rss=psutil.Process().memory_info().rss/2**30
    if rss>c["max_process_gb"] or psutil.virtual_memory().available/2**30<c["min_available_gb"]:
        raise MemoryError("Close other kernels or reduce donor/focal sample and batch size")
    return rss


def exact_id(s,kind=None,allow_missing=False):
    if pd.api.types.is_float_dtype(s.dtype) or pd.api.types.is_bool_dtype(s.dtype):
        raise ValueError("Identifiers must never pass through floating point or bool")
    x=s.astype("string")
    missing=x.isna()
    if missing.any() and not allow_missing:raise ValueError("Missing identifier")
    pattern=(r"(?:https?://openalex\.org/)?"+kind+r"?([0-9]+)") if kind else r"([0-9]+)"
    out=x.str.extract("^"+pattern+"$",expand=False)
    if (out.isna()&~missing).any():raise ValueError("Invalid exact identifier")
    return out.astype("UInt64") if allow_missing else out.astype("uint64")


def hash64(ids,seed):
    x=np.asarray(ids,dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        x+=np.uint64(seed)+np.uint64(0x9E3779B97F4A7C15)
        x=(x^(x>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9)
        x=(x^(x>>np.uint64(27)))*np.uint64(0x94D049BB133111EB)
        x^=x>>np.uint64(31)
    return x


def batches(path,columns,c):
    pf=pq.ParquetFile(path)
    missing=set(columns)-set(pf.schema_arrow.names)
    if missing:raise KeyError(f"{path}: missing {sorted(missing)}")
    for b in pf.iter_batches(batch_size=c["batch_rows"],columns=columns):
        yield b.to_pandas(types_mapper=pd.ArrowDtype)
        guard(c)


def write_frame(d,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".partial")
    d.to_parquet(tmp,index=False,compression="zstd")
    os.replace(tmp,path)


class Sinks:
    """One bounded writer per annual / bucket shard; pandas does transforms."""
    def __init__(self,directory,key="year"):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.key=key;self.writers={}
    def add(self,d):
        for key,part in d.groupby(self.key,sort=False,observed=True):
            key=int(key);part=part.drop(columns=self.key).reset_index(drop=True)
            if part.empty:continue
            table=pa.Table.from_pandas(part,preserve_index=False)
            if key not in self.writers:
                self.writers[key]=pq.ParquetWriter(self.directory/f"{key}.parquet",table.schema,compression="zstd")
            self.writers[key].write_table(table)
    def close(self):
        for w in self.writers.values():w.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def stage(c,name,fn):
    out=Path(c["_out"]);marker=out/"cache"/(name+".ready")
    if marker.exists():return
    print("Preparing",name,flush=True);fn(c);marker.write_text("complete\n")
    gc.collect();guard(c)


def core_index(c):
    directory=Path(c["_out"])/"cache/core_index";directory.mkdir(parents=True,exist_ok=True)
    arrays={k:np.load(directory/(k+".npy"),mmap_mode="r") for k in ["paper_id","year","source_code","field_id","source_ids"]}
    arrays["index"]=pd.Index(arrays["paper_id"])
    return arrays


def build_core(c):
    path=Path(c["_data_root"])/"core/papers.parquet"
    cols=["paper_id","year","source_id","primary_field_id"]
    frames=[]
    for d in batches(path,cols,c):
        d=d.loc[d.year.le(c["year_end"])].copy()
        for v in ["paper_id","source_id","primary_field_id"]:
            d[v]=exact_id(d[v],allow_missing=v!="paper_id")
        frames.append(d)
    d=pd.concat(frames,ignore_index=True).sort_values("paper_id",kind="stable")
    if d.paper_id.duplicated().any():raise ValueError("Duplicate core paper ID")
    sources=np.sort(d.source_id.dropna().loc[lambda s:s.gt(0)].unique().astype(np.uint64))
    si=pd.Index(sources)
    code=np.full(len(d),-1,dtype=np.int32);known=d.source_id.notna()&d.source_id.gt(0)
    code[known]=si.get_indexer(d.loc[known,"source_id"].astype(np.uint64)).astype(np.int32)
    directory=Path(c["_out"])/"cache/core_index";directory.mkdir(parents=True,exist_ok=True)
    for name,values in dict(paper_id=d.paper_id.to_numpy(np.uint64),year=d.year.to_numpy(np.int16),
                            field_id=d.primary_field_id.fillna(0).to_numpy(np.uint64),
                            source_code=code,source_ids=sources).items():
        np.save(directory/(name+".npy"),values)


def focal_sample(c):
    root=Path(c["_analysis_run"])
    if not (root/"DATA_READY").exists():raise FileNotFoundError("Run 01 or set analysis_run_dir to an existing DATA_READY run")
    frames=[]
    for p in sorted((root/"data").glob("*.parquet")):
        available=pq.ParquetFile(p).schema_arrow.names
        missing=set(BASE_COLUMNS)-set(available)
        if missing:raise KeyError(f"{p}: {sorted(missing)}")
        d=pd.read_parquet(p,columns=BASE_COLUMNS)
        d=d[d.year.between(c["year_start"],c["year_end"])]
        frames.append(d)
    if not frames:raise ValueError("No annual analysis files")
    d=pd.concat(frames,ignore_index=True)
    d["paper_id"]=exact_id(d.paper_id)
    if d.paper_id.duplicated().any():raise ValueError("Duplicate focal papers")
    mode=c.get("focal_sampling","paper_hash")
    if mode=="reference_member_clusters":
        refs=exact_id(d.reference_member_id)
        threshold=int(c["reference_member_fraction"]*(2**64-1))
        keep=hash64(refs,c["focal_seed"])<np.uint64(threshold) if c["reference_member_fraction"]<1 else np.ones(len(d),bool)
        d=d.loc[keep].copy()
    elif mode=="paper_hash":
        h=hash64(d.paper_id,c["focal_seed"])
        order=np.lexsort((d.paper_id.to_numpy(),h))
        d=d.iloc[order[:c["max_focal_papers"]]]
    else:raise ValueError("Unknown focal_sampling")
    d=d.sort_values(["year","paper_id"]).reset_index(drop=True)
    if d.empty:raise ValueError("No focal clusters selected")
    print("Focal papers",len(d),"reference-member clusters",d.reference_member_id.nunique(),flush=True)
    write_frame(d,Path(c["_out"])/"focal_panel.parquet")
    wanted=pd.Index(d.paper_id)
    sinkdir=Path(c["_out"])/"cache/slots_input"
    cached=Path(c["_analysis_run"])/"cache/pivot"
    files=sorted(cached.glob("*.parquet")) if cached.exists() else []
    files=files or [Path(c["_data_root"])/"metrics/author_paper_pivot.parquet"]
    with Sinks(sinkdir) as sink:
        for p in files:
            for z in batches(p,PIVOT_COLUMNS,c):
                z["paper_id"]=exact_id(z.paper_id);z["author_id"]=exact_id(z.author_id)
                z=z[z.paper_id.isin(wanted)].copy()
                if len(z):sink.add(z)
    pieces=[pd.read_parquet(p).assign(year=int(p.stem)) for p in sinkdir.glob("*.parquet")]
    slots=pd.concat(pieces,ignore_index=True)
    if slots.duplicated(["paper_id","author_id"]).any():raise ValueError("Duplicate focal author relation")
    counts=slots.groupby("paper_id").size().reindex(d.paper_id)
    if not np.array_equal(counts.to_numpy(),d.team_size.to_numpy()):raise ValueError("Incomplete focal member lists")


def active_pool(c):
    core=core_index(c);directory=Path(c["_out"])/"cache/active_sample"
    path=Path(c["raw_root"])/c["raw_authors_file"]
    with Sinks(directory) as sink:
        for z in batches(path,["paperid","authorid"],c):
            p=exact_id(z.paperid,"W").to_numpy();a=exact_id(z.authorid,"A").to_numpy()
            pos=core["index"].get_indexer(p);ok=pos>=0
            p,a,pos=p[ok],a[ok],pos[ok];year=core["year"][pos]
            ok=(year>=c["year_start"])&(year<=c["year_end"])
            a,year=a[ok],year[ok]
            parts=[]
            for y in np.unique(year):
                ids=np.unique(a[year==y]);hv=hash64(ids,c["seed"]+int(y))
                keep=hv<np.uint64(int(c["donor_fraction"]*(2**64-1))) if c["donor_fraction"]<1 else np.ones(len(ids),bool)
                if keep.any():parts.append(pd.DataFrame({"author_id":ids[keep],"year":np.int16(y)}))
            if parts:sink.add(pd.concat(parts,ignore_index=True))
    pooldir=Path(c["_out"])/"cache/pool";pooldir.mkdir(parents=True,exist_ok=True)
    for slotfile in sorted((Path(c["_out"])/"cache/slots_input").glob("*.parquet")):
        y=int(slotfile.stem);forced=pd.read_parquet(slotfile,columns=["author_id"]).author_id.unique()
        sampled=directory/f"{y}.parquet"
        ids=pd.read_parquet(sampled).author_id.unique() if sampled.exists() else np.empty(0,np.uint64)
        ids=np.setdiff1d(ids,forced);order=np.argsort(hash64(ids,c["seed"]+y),kind="stable")
        ids=np.sort(np.union1d(forced,ids[order[:c["max_donors_per_year"]]]))
        write_frame(pd.DataFrame({"author_id":ids}),pooldir/f"{y}.parquet")


def history_relations(c):
    core=core_index(c);L=c["history_years"]
    pools={int(p.stem):pd.read_parquet(p).author_id.to_numpy(np.uint64)
           for p in (Path(c["_out"])/"cache/pool").glob("*.parquet")}
    union=pd.Index(np.unique(np.concatenate(list(pools.values()))))
    memberships={y:pd.Index(a) for y,a in pools.items()}
    path=Path(c["raw_root"])/c["raw_authors_file"]
    with Sinks(Path(c["_out"])/"cache/history_relations") as sink:
        for z in batches(path,["paperid","authorid"],c):
            a=exact_id(z.authorid,"A").to_numpy();ok=union.get_indexer(a)>=0
            if not ok.any():continue
            a=a[ok];p=exact_id(z.loc[ok,"paperid"],"W").to_numpy()
            pos=core["index"].get_indexer(p);ok=pos>=0
            a,p,pos=a[ok],p[ok],pos[ok];years=core["year"][pos]
            pieces=[]
            for lag in range(1,L+1):
                target=years+lag
                for y in np.unique(target):
                    if int(y) not in memberships:continue
                    mask=target==y;aa=a[mask];pp=pos[mask]
                    keep=memberships[int(y)].get_indexer(aa)>=0
                    if keep.any():
                        pieces.append(pd.DataFrame({"author_id":aa[keep],"paper_pos":pp[keep].astype(np.int64),"year":np.int16(y)}))
            if pieces:sink.add(pd.concat(pieces,ignore_index=True))
    # Author-year covariates are inherited prepublication states.
    with Sinks(Path(c["_out"])/"cache/author_state") as sink:
        for z in batches(Path(c["_data_root"])/"core/authors_year.parquet",
                         ["author_id","year","prior_papers","prior_h","career_age"],c):
            z["author_id"]=exact_id(z.author_id)
            pieces=[]
            for y,part in z.groupby("year",sort=False):
                if int(y) not in memberships:continue
                part=part[part.author_id.isin(memberships[int(y)])]
                if len(part):pieces.append(part)
            if pieces:sink.add(pd.concat(pieces,ignore_index=True))


def reference_profiles(c):
    out=Path(c["_out"]);core=core_index(c);n=len(core["paper_id"])
    wanted=np.zeros(n,bool)
    for p in (out/"cache/history_relations").glob("*.parquet"):
        for d in batches(p,["paper_pos"],c):wanted[d.paper_pos.to_numpy(np.int64)]=True
    panel=pd.read_parquet(out/"focal_panel.parquet",columns=["paper_id"])
    pos=core["index"].get_indexer(panel.paper_id)
    if np.any(pos<0):raise ValueError("Focal/core versions differ")
    wanted[pos]=True
    # Only selected history/current papers' edges are cached. Citation targets
    # still range over the entire research core, including older publications.
    edge_ready=out/"cache/reference_edges.ready"
    if not edge_ready.exists():
        edge_directory=out/"cache/reference_edges"
        if edge_directory.exists():shutil.rmtree(edge_directory)
        with Sinks(edge_directory) as sink:
            for z in batches(Path(c["raw_root"])/c["raw_references_file"],
                             ["citing_paperid","cited_paperid"],c):
                a=core["index"].get_indexer(exact_id(z.citing_paperid,"W"))
                b=core["index"].get_indexer(exact_id(z.cited_paperid,"W"))
                ok=(a>=0)&(b>=0);a,b=a[ok],b[ok]
                ok=wanted[a]&(a!=b)&(core["year"][b]<=core["year"][a])&(core["source_code"][b]>=0)
                a,b=a[ok],b[ok]
                if len(a):sink.add(pd.DataFrame({"citing_pos":a.astype(np.int64),"cited_pos":b.astype(np.int64),"year":core["year"][a].astype(np.int16)}))
        edge_ready.write_text("ready\n")
    profile_dir=out/"cache/paper_profiles";profile_dir.mkdir(parents=True,exist_ok=True)
    for y in sorted(np.unique(core["year"][wanted])):
        year_ready=profile_dir/f"{int(y)}.ready"
        if year_ready.exists():continue
        rowpos=np.flatnonzero(wanted&(core["year"]==y));rowindex=pd.Index(rowpos)
        path=out/"cache/reference_edges"/f"{int(y)}.parquet"
        grouped=[]
        if path.exists():
            bucketdir=out/"cache/reference_buckets"/str(int(y))
            with Sinks(bucketdir,key="bucket") as sink:
                for z in batches(path,["citing_pos","cited_pos"],c):
                    z["bucket"]=(z.citing_pos.to_numpy(np.int64)%c["reference_buckets"]).astype(np.int16);sink.add(z)
            for p in sorted(bucketdir.glob("*.parquet")):
                z=pd.read_parquet(p).drop_duplicates(["citing_pos","cited_pos"])
                z["source_code"]=core["source_code"][z.cited_pos.to_numpy(np.int64)]
                z=z.groupby(["citing_pos","source_code"],sort=False).size().rename("count").reset_index()
                grouped.append(z);guard(c)
        if grouped:
            z=pd.concat(grouped,ignore_index=True)
            X=sp.coo_matrix((z["count"].to_numpy(float),(rowindex.get_indexer(z.citing_pos),z.source_code.to_numpy(np.int32))),
                            shape=(len(rowpos),len(core["source_ids"]))).tocsr()
        else:X=sp.csr_matrix((len(rowpos),len(core["source_ids"])),dtype=float)
        np.save(profile_dir/f"{int(y)}_rows.npy",rowpos.astype(np.int64))
        sp.save_npz(profile_dir/f"{int(y)}.npz",X)
        year_ready.write_text("ready\n")
        bucketdir=out/"cache/reference_buckets"/str(int(y))
        if bucketdir.exists():shutil.rmtree(bucketdir)
        if not c.get("keep_reference_edges",True) and path.exists():path.unlink()
        print("Source profiles",int(y),X.shape,X.nnz,flush=True)
        del X,grouped;gc.collect();guard(c)


def cosine_slots(X,H,paper_rows,history_rows,batch_rows=25000,norms=None):
    result=np.empty(len(paper_rows),float)
    if norms is None:
        xn=np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        hn=np.sqrt(np.asarray(H.multiply(H).sum(axis=1)).ravel())
    else:xn,hn=norms
    for start in range(0,len(result),batch_rows):
        sl=slice(start,min(start+batch_rows,len(result)));p=paper_rows[sl];h=history_rows[sl]
        numerator=np.asarray(X[p].multiply(H[h]).sum(axis=1)).ravel()
        den=xn[p]*hn[h]
        if np.any(den<=0):raise ValueError("Zero-norm profile in scored member slots")
        result[sl]=np.clip(1-numerator/den,0,1)
    return result


def aggregate_configuration(distances,paper_rows,n_papers,team_sizes):
    count=np.bincount(paper_rows,minlength=n_papers)
    if not np.array_equal(count,np.asarray(team_sizes)):raise ValueError("A complete team is required")
    mean=np.bincount(paper_rows,weights=distances,minlength=n_papers)/count
    minimum=np.full(n_papers,np.inf);np.minimum.at(minimum,paper_rows,distances)
    return mean,mean-minimum


def history_descriptors(H,pool):
    """Lightweight strata from existing history counts; no focal outcomes."""
    pool=pool.copy();total=np.asarray(H.sum(axis=1)).ravel()
    peak=np.asarray(H.max(axis=1).toarray()).ravel()
    pool["dominant_source_code"]=np.asarray(H.argmax(axis=1)).ravel().astype(np.int32)
    pool["dominant_source_share"]=np.divide(peak,total,out=np.zeros(len(total)),where=total>0)
    pool["history_source_count"]=np.diff(H.indptr).astype(np.int32)
    return pool


def ensure_history_descriptors(c):
    for directory in sorted((Path(c["_out"])/"cache/years").iterdir()):
        pool=pd.read_parquet(directory/"pool.parquet")
        if "dominant_source_code" not in pool:
            H=sp.load_npz(directory/"histories.npz")
            write_frame(history_descriptors(H,pool),directory/"pool.parquet")
            del H;gc.collect();guard(c)


def assemble_years(c):
    out=Path(c["_out"]);core=core_index(c)
    panel=pd.read_parquet(out/"focal_panel.parquet")
    for p in sorted((out/"cache/slots_input").glob("*.parquet")):
        y=int(p.stem);directory=out/"cache/years"/str(y);directory.mkdir(parents=True,exist_ok=True)
        if (directory/"YEAR_READY").exists():continue
        state=pd.read_parquet(out/"cache/author_state"/p.name)
        if state.author_id.duplicated().any():raise ValueError("Duplicate author/year state")
        pool=pd.read_parquet(out/"cache/pool"/p.name).sort_values("author_id").reset_index(drop=True)
        pool=pool.merge(state,on="author_id",how="left",validate="one_to_one")
        pool["history_row"]=np.arange(len(pool),dtype=np.int64);aindex=pd.Index(pool.author_id)
        rel=pd.read_parquet(out/"cache/history_relations"/p.name).drop_duplicates(["author_id","paper_pos"])
        matrices=[];histrows=[]
        for yy in range(y-c["history_years"],y):
            fn=out/"cache/paper_profiles"/f"{yy}.npz"
            if fn.exists():
                matrices.append(sp.load_npz(fn));histrows.append(np.load(fn.with_name(f"{yy}_rows.npy")))
        if not matrices:raise ValueError(f"{y}: no historical source profiles")
        Xh=sp.vstack(matrices,format="csr");hp=np.concatenate(histrows);hindex=pd.Index(hp)
        col=hindex.get_indexer(rel.paper_pos)
        if np.any(col<0):raise ValueError("Historical source profile missing")
        A=sp.coo_matrix((np.ones(len(rel)),(aindex.get_indexer(rel.author_id),col)),shape=(len(pool),len(hp))).tocsr()
        H=(A@Xh).tocsr()
        pool["history_refs"]=np.asarray(H.sum(axis=1)).ravel()
        pool=history_descriptors(H,pool)
        rel["field_id"]=core["field_id"][rel.paper_pos.to_numpy(np.int64)]
        freq=rel[rel.field_id.gt(0)].groupby(["author_id","field_id"],sort=False).size().rename("n").reset_index()
        dominant=freq.sort_values(["author_id","n","field_id"],ascending=[True,False,True],kind="stable").drop_duplicates("author_id")
        dominant["field_id"]=dominant.field_id.astype("UInt64")
        pool=pool.merge(dominant[["author_id","field_id"]],on="author_id",how="left",validate="one_to_one")
        pool["field_id"]=pool.field_id.astype("UInt64")
        pool["eligible"]=(pool.history_refs.ge(c["min_history_references"])&
             pool.career_age.between(c["career_age_min"],c["career_age_max"])&pool.field_id.notna())
        yp=panel[panel.year.eq(y)].sort_values("paper_id").reset_index(drop=True)
        source_rows=np.load(out/"cache/paper_profiles"/f"{y}_rows.npy")
        source_matrix=sp.load_npz(out/"cache/paper_profiles"/f"{y}.npz")
        global_pos=core["index"].get_indexer(yp.paper_id)
        prow=pd.Index(source_rows).get_indexer(global_pos)
        if np.any(prow<0):raise ValueError("Current source profile missing")
        X=source_matrix[prow].tocsr()
        slots=pd.read_parquet(p).sort_values(["paper_id","author_id"]).reset_index(drop=True)
        slots["paper_row"]=pd.Index(yp.paper_id).get_indexer(slots.paper_id)
        slots["history_row"]=aindex.get_indexer(slots.author_id)
        if (slots.history_row<0).any():raise ValueError("Focal author missing from donor population")
        observed=cosine_slots(X,H,slots.paper_row.to_numpy(),slots.history_row.to_numpy(),c["slot_batch_rows"])
        old=slots.reference_source_pivot.to_numpy(float)
        if not np.isfinite(old).all():raise ValueError("Invalid original focal member distances")
        error=float(np.max(np.abs(observed-old))) if len(old) else 0
        hcount=pool.history_refs.to_numpy()[slots.history_row]
        xcount=np.asarray(X.sum(axis=1)).ravel()[slots.paper_row]
        if not np.allclose(hcount,slots.prior_window_reference_count,atol=1e-6,rtol=0):
            raise ValueError(f"{y}: reconstructed history counts differ; verify the exact raw/core snapshot and pivot window")
        if not np.allclose(xcount,slots.current_reference_count,atol=1e-6,rtol=0):
            raise ValueError(f"{y}: reconstructed current reference counts differ")
        if error>c["reconstruction_atol"]:raise ValueError(f"{y}: pivot reconstruction max error {error}; benchmark stopped")
        M,G=aggregate_configuration(observed,slots.paper_row.to_numpy(),len(yp),yp.team_size)
        if not np.allclose(M,yp.mean_pivot,atol=c["reconstruction_atol"],rtol=0) or not np.allclose(G,yp.grounding_gap,atol=2*c["reconstruction_atol"],rtol=0):
            raise ValueError(f"{y}: reconstructed team coordinates disagree")
        yp["original_mean_pivot"]=yp.mean_pivot;yp["original_grounding_gap"]=yp.grounding_gap
        yp["mean_pivot"]=M;yp["grounding_gap"]=G
        slots["reconstructed_pivot"]=observed
        sp.save_npz(directory/"histories.npz",H);sp.save_npz(directory/"current.npz",X)
        write_frame(pool,directory/"pool.parquet");write_frame(slots,directory/"slots.parquet")
        write_frame(yp,directory/"papers.parquet")
        (directory/"YEAR_READY").write_text("ready\n")
        del H,X,Xh,A,source_matrix;gc.collect();guard(c)


def prepare(c=None):
    c=c or config();out=Path(c["_out"]);(out/"cache").mkdir(parents=True,exist_ok=True)
    required=[Path(c["_data_root"])/"core/papers.parquet",
              Path(c["_data_root"])/"core/authors_year.parquet",
              Path(c["raw_root"])/c["raw_authors_file"],Path(c["raw_root"])/c["raw_references_file"]]
    for p in required:
        if not p.exists():raise FileNotFoundError(f"Required graph input: {p}. Scalar pivot tables cannot replace this input.")
    keys=["_data_root","_analysis_run","raw_root","raw_authors_file","raw_references_file",
          "year_start","year_end","history_years","max_focal_papers","focal_seed","donor_fraction",
          "max_donors_per_year","min_history_references","career_age_min","career_age_max","seed","reference_buckets",
          "focal_sampling","reference_member_fraction"]
    contract={k:c.get(k) for k in keys}
    annual=sorted((Path(c["_analysis_run"])/"data").glob("*.parquet"))
    pivots=sorted((Path(c["_analysis_run"])/"cache/pivot").glob("*.parquet"))
    pivots=pivots or [Path(c["_data_root"])/"metrics/author_paper_pivot.parquet"]
    contract["files"]={str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in required+annual+pivots}
    settings_path=out/"profile_settings.json"
    if settings_path.exists():
        previous=json.loads(settings_path.read_text())
        if previous!=contract:
            changed=sorted(k for k in set(previous)|set(contract) if previous.get(k)!=contract.get(k))
            completed=(any((out/"cache").glob("*.ready")) or (out/"PROFILES_READY").exists()
                       or any(out.glob("null/*/SCENARIO_READY")) or any(out.glob("cache/years/*/YEAR_READY")))
            if completed:
                raise ValueError("Cached source profiles use different inputs/settings: "+", ".join(changed)+
                    ". Set a new benchmark run_name; existing notebook 01/02 outputs can be reused.")
            # No completed stage exists: discard only this run's unfinished
            # cache, never the separate original annual analysis data.
            shutil.rmtree(out/"cache");(out/"cache").mkdir()
    atomic_json(settings_path,contract)
    for name,fn in [("focal",focal_sample),("core",build_core),("active_pool",active_pool),
                    ("history_relations",history_relations),("paper_profiles",reference_profiles),
                    ("year_profiles",assemble_years)]:stage(c,name,fn)
    (out/"PROFILES_READY").write_text("ready\n")
    activate(c)
    return out


if __name__=="__main__":prepare()
