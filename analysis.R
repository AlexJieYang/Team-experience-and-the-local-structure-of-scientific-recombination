suppressPackageStartupMessages({library(fixest);library(data.table);library(arrow);library(jsonlite)})
main_terms<-c('m','g','m2','g2','mg')
controls<-c('log_refs','log_team_h','log_team_output','log_ref_output','log_ref_h')
base_fe<-'reference_member_id + primary_field_id^year^is_conference + team_size'
venue_fe<-paste(base_fe,'source_id^year',sep=' + ')

init<-function(project=getwd()) {
 cfg<-fromJSON(file.path(project,'config.json'));out<-file.path(project,'outputs',cfg$run_name)
 stopifnot(file.exists(file.path(out,'DATA_READY')))
 if(!identical(cfg,fromJSON(file.path(out,'config_used.json'))))stop('Configuration does not match prepared data')
 setFixest_nthreads(cfg$r_threads);setDTthreads(cfg$r_threads)
 list(cfg=cfg,out=out)
}
features<-function(m,g,spline=FALSE) {
 if(!spline)return(data.frame(m=m-.5,g=g-.2,m2=(m-.5)^2,g2=(g-.2)^2,mg=(m-.5)*(g-.2)))
 bm<-splines::ns(m,knots=c(1/3,2/3),Boundary.knots=c(0,1))
 bg<-splines::ns(g,knots=c(1/3,2/3),Boundary.knots=c(0,1))
 x<-as.data.frame(cbind(bm,bg));names(x)<-c(paste0('bm',1:3),paste0('bg',1:3))
 for(i in 1:3)for(j in 1:3)x[[paste0('t',i,j)]]<-bm[,i]*bg[,j]
 x
}
load_data<-function(ctx) {
 a<-fread(file.path(ctx$out,'tables/annual_sample.csv'))
 if(sum(a$papers)>ctx$cfg$r_max_rows)stop('Too many rows for configured R limit; reduce paper_fraction in a new run')
 cols<-unique(c('paper_id','reference_member_id','h_reference_id','source_id','primary_field_id','year','is_conference',
  'mean_pivot','grounding_gap','q25_gap','pivot_sd','history_coverage','team_size','citation_percentile',
  'novel_share_pct','any_novel','any_novel_pct','joint_success','joint_success_pct','high_recognition',
  'positive_novel_share','high_recognition_pct','high75_pct','high95_pct','joint75_pct','joint95_pct','c_total','log_citations','semantic_distance','z10_asinh','z50_asinh',
  'split_half','prior_h_gini','prior_output_log_sd','log_min_history_refs','log_mean_history_refs',
  'n_reference_sources','source_hhi_pct','source_count_bin','hhi_bin','ref_count_bin','log_n_sources','reference_source_hhi','mean_ref_age','log_ref_citation_base','team_between_js','team_within_shannon','prior_team_edge_density','log_degree','ref_field_rao',controls))
 paths<-sort(list.files(file.path(ctx$out,'data'),pattern='[.]parquet$',full.names=TRUE))
 d<-rbindlist(lapply(paths,function(p)as.data.table(read_parquet(p,col_select=tidyselect::all_of(cols)))))
 stopifnot(is.character(d$paper_id),!anyDuplicated(d$paper_id))
 for(v in c('paper_id','reference_member_id','h_reference_id','source_id','primary_field_id')) set(d,j=v,value=as.character(d[[v]]))
 for(v in names(d)) {
  if(inherits(d[[v]],'integer64'))set(d,j=v,value=as.numeric(d[[v]]))
  if(is.numeric(d[[v]]))set(d,which(!is.finite(d[[v]])),v,NA)
 }
 d[,field_id_text:=as.character(primary_field_id)]
 for(v in c('reference_member_id','h_reference_id','source_id','primary_field_id')) {
  vals<-unique(d[[v]][!is.na(d[[v]])]);set(d,j=v,value=match(d[[v]],vals))
 }
 # paper_id stays exact text; no clustering by a unique paper label is needed.
 f<-features(d$mean_pivot,d$grounding_gap)
 for(v in names(f))set(d,j=v,value=f[[v]])
 if(as.numeric(object.size(d))/1024^3>ctx$cfg$max_process_gb/4)stop('Insufficient memory headroom')
 d
}
compact_model<-function(m) {
 m<-summary(m,lean=TRUE);m$call_env<-NULL;m$fixest_env<-NULL
 f<-m$fml_all$linear;environment(f)<-baseenv()
 m$call<-as.call(list(as.name('feols'),fml=f))
 clean<-function(x) {
  if(inherits(x,'formula')){environment(x)<-baseenv();return(x)}
  if(is.environment(x))return(baseenv())
  if(is.list(x))for(i in seq_along(x))x[i]<-list(clean(x[[i]]))
  x
 }
 clean(m)
}
write_rds<-function(x,path) {
 tmp<-tempfile('.writing_',tmpdir=dirname(path));on.exit(unlink(tmp),add=TRUE)
 saveRDS(x,tmp,compress='gzip')
 if(file.exists(path)&&!file.remove(path))stop('Cannot replace checkpoint file')
 if(!file.rename(tmp,path))stop('Cannot finalize checkpoint file')
}
registry<-function(){e<-new.env(parent=emptyenv());e$results<-list();e$field_coverage<-list();e$model_status<-list();e}

support_counts<-function(d,grid) {
 tab<-table(factor(pmin(floor(d$mean_pivot*20),19),levels=0:19),factor(pmin(floor(d$grounding_gap*20),19),levels=0:19))
 as.integer(tab[cbind(pmin(floor(grid$mean_pivot*20),19)+1,pmin(floor(grid$grounding_gap*20),19)+1)])
}
contrast_matrix<-function(b,x,x0) {
 L<-matrix(0,nrow(x),length(b),dimnames=list(NULL,names(b)))
 for(v in intersect(names(x),names(b)))L[,v]<-x[[v]]-x0[[v]][1]
 L
}
model_geometry<-function(m,d,ctx,name,y,spline=FALSE,conditional_support=FALSE) {
 grid<-expand.grid(mean_pivot=seq(.025,.975,.05),grounding_gap=seq(.025,.975,.05))
 grid$n_cell<-support_counts(d,grid)
 grid$supported<-grid$grounding_gap<=grid$mean_pivot & grid$n_cell>=ctx$cfg$surface_cell_min
 ref<-c(.6,.2)
 if(support_counts(d,data.frame(mean_pivot=ref[1],grounding_gap=ref[2]))<ctx$cfg$surface_cell_min) {
  k<-which.max(grid$n_cell);ref<-c(grid$mean_pivot[k],grid$grounding_gap[k])
 }
 # Enforce at least one adequately represented exact team size at BOTH the
 # reference and evaluated location, beyond pooled geometric support.
 by_size<-table(factor(pmin(floor(d$mean_pivot*20),19),levels=0:19),
                factor(pmin(floor(d$grounding_gap*20),19),levels=0:19),factor(d$team_size))
 min_size_n<-max(5L,min(20L,as.integer(ctx$cfg$surface_cell_min/5)))
 ref_sizes<-by_size[min(floor(ref[1]*20),19)+1,min(floor(ref[2]*20),19)+1,]>=min_size_n
 grid$common_size_support<-vapply(seq_len(nrow(grid)),function(k) {
  cc<-by_size[min(floor(grid$mean_pivot[k]*20),19)+1,min(floor(grid$grounding_gap[k]*20),19)+1,]
  any(cc>=min_size_n & ref_sizes)
 },logical(1))
 grid$supported<-grid$supported & grid$common_size_support
 context_data<-context_index(d,ctx$cfg)
 context<-context_support(d,grid,ref,ctx$cfg,index=context_data)
 grid[,c('context_cells','context_mass_here','context_mass_reference')]<-context
 if(conditional_support)grid$supported<-grid$supported & context$context_cells>0L & context$context_mass_here>=ctx$cfg$profile_common_mass_min & context$context_mass_reference>=ctx$cfg$profile_common_mass_min
 b<-coef(m);V<-vcov(m);x<-features(grid$mean_pivot,grid$grounding_gap,spline)
 L<-contrast_matrix(b,x,features(ref[1],ref[2],spline))
 grid$contrast<-as.numeric(L%*%b);grid$se<-sqrt(pmax(0,rowSums((L%*%V)*L)))
 grid$lower<-grid$contrast-1.96*grid$se;grid$upper<-grid$contrast+1.96*grid$se
 dx<-(as.matrix(features(grid$mean_pivot,grid$grounding_gap+1e-5,spline))-
       as.matrix(features(grid$mean_pivot,grid$grounding_gap-1e-5,spline)))/(2e-5)
 D<-L*0;for(v in intersect(colnames(dx),names(b)))D[,v]<-.1*dx[,v]
 grid$gap_slope<-as.numeric(D%*%b);grid$gap_slope_se<-sqrt(pmax(0,rowSums((D%*%V)*D)))
 grid$model<-name;grid$outcome<-y;grid$ref_mean<-ref[1];grid$ref_gap<-ref[2]
 # Fixed, interpretable contrasts: same mean, gap increases from .05 to .25.
 contrasts<-list();checks<-list()
 for(av in c(.4,.6,.8)) {
  points<-data.frame(mean_pivot=c(av,av),grounding_gap=c(.05,.25))
  density_n<-support_counts(d,points)
  density_ok<-all(density_n>=ctx$cfg$surface_cell_min)
  sf<-by_size[min(floor(av*20),19)+1,2,]
  st<-by_size[min(floor(av*20),19)+1,6,]
  common_sizes<-dimnames(by_size)[[3]][sf>=min_size_n & st>=min_size_n]
  cc<-context_support(d,points[2,,drop=FALSE],c(av,.05),ctx$cfg,index=context_data)
  context_ok<-cc$context_cells[1]>0L && cc$context_mass_here[1]>=ctx$cfg$profile_common_mass_min && cc$context_mass_reference[1]>=ctx$cfg$profile_common_mass_min
  eligible<-density_ok && length(common_sizes)>0L && (!conditional_support || context_ok)
  checks[[length(checks)+1L]]<-data.table(model=name,outcome=y,mean_pivot=av,gap_from=.05,gap_to=.25,
    n_from=density_n[1],n_to=density_n[2],common_team_sizes=paste(common_sizes,collapse=','),
    context_cells=cc$context_cells[1],context_mass_here=cc$context_mass_here[1],context_mass_reference=cc$context_mass_reference[1],
    supported=eligible,reason=if(!density_ok)'sparse M/G endpoints' else if(!length(common_sizes))'no common exact team size' else
      if(conditional_support && !context_ok)'insufficient common reference contexts' else '')
  if(!eligible)next
  a<-features(av,.25,spline);a0<-features(av,.05,spline);ll<-contrast_matrix(b,a,a0)
  est<-as.numeric(ll%*%b);se<-sqrt(max(0,as.numeric(ll%*%V%*%t(ll))))
  contrasts[[length(contrasts)+1L]]<-data.table(model=name,outcome=y,mean_pivot=av,
       gap_from=.05,gap_to=.25,estimate=est,se=se,lower=est-1.96*se,upper=est+1.96*se,
       p=if(se>0)2*pnorm(-abs(est/se)) else NA_real_,
       common_team_sizes=paste(common_sizes,collapse=','),
       context_cells=cc$context_cells[1],context_mass_here=cc$context_mass_here[1],context_mass_reference=cc$context_mass_reference[1],conditional_support=conditional_support)
 }
 empty<-data.table(model=character(),outcome=character(),mean_pivot=numeric(),gap_from=numeric(),gap_to=numeric(),
                   estimate=numeric(),se=numeric(),lower=numeric(),upper=numeric(),p=numeric())
 list(surface=as.data.table(grid),contrasts=if(length(contrasts))rbindlist(contrasts,fill=TRUE) else empty,support=rbindlist(checks,fill=TRUE))
}
fit_impl<-function(e,ctx,d,name,y,terms=main_terms,ctrl=controls,fe=base_fe,spline=FALSE,
              surface=TRUE,poisson=FALSE,keep_rows=FALSE,required=TRUE,cluster_ref='reference_member_id',conditional_support=FALSE) {
 formula<-as.formula(paste(y,'~',paste(c(terms,ctrl),collapse=' + '),'|',fe),env=baseenv())
 spec<-paste(paste(deparse(formula),collapse=' '),if(poisson)'PPML' else 'OLS',cluster_ref,sep=' :: ')
 p<-file.path(ctx$out,'models',paste0(name,'.rds'))
 vars<-unique(c(all.vars(formula),cluster_ref,'source_id','year'))
 used<-which(complete.cases(d[,..vars]))
 if(length(used)<500L) {
  if(required)stop(paste(name,'has fewer than 500 complete papers'))
  cat(name,': insufficient sample; not estimated\n');return(NULL)
 }
 args<-list(fml=formula,data=d[used],vcov=as.formula(paste('~',cluster_ref,'+ source_id^year'),env=baseenv()),
            lean=FALSE,mem.clean=TRUE,nthreads=ctx$cfg$r_threads)
 m<-do.call(if(poisson)fepois else feols,args)
 if(identical(m$convStatus,FALSE))stop(paste(name,'did not converge'))
 if(surface && length(setdiff(terms,names(coef(m)))))
  stop(paste(name,'has unidentified surface terms:',paste(setdiff(terms,names(coef(m))),collapse=', ')))
 actual<-used[fixest::obs(m)];stopifnot(length(actual)==nobs(m))
 m<-compact_model(m);rm(args);gc()
 z<-as.data.frame(coeftable(m))
 tab<-data.table(model=name,outcome=y,term=rownames(z),estimate=z[[1]],se=z[[2]],p=z[[4]])
 wt<-tryCatch(wald(m,keep=if(spline)'^bg|^t[1-3]' else '^g$|^g2$|^mg$',print=FALSE),error=function(err)NA)
 test<-if(length(wt)>1)data.table(model=name,outcome=y,stat=unname(wt['stat']),p=unname(wt['p'])) else NULL
 geometry<-if(surface&&!poisson)model_geometry(m,d[actual],ctx,name,y,spline,conditional_support) else list(surface=NULL,contrasts=NULL)
 bundle<-list(spec=spec,n_input=nrow(d),n_complete=length(used),n_estimated=nobs(m),
              clusters_reference=uniqueN(d[[cluster_ref]][actual]),
              clusters_source_year=uniqueN(d[actual,.(source_id,year)]),
              coefficients=tab,vcov=vcov(m),test=test,surface=geometry$surface,contrasts=geometry$contrasts,profile_support=geometry$support,
              rows=if(keep_rows)actual else NULL)
 # Bundle contains summaries and at most required row indices: never a fit environment or input data.
 e$results[[name]]<-bundle
 tryCatch(write_rds(bundle,p),error=function(err){
  warning(paste('Model remains in memory; checkpoint write failed:',conditionMessage(err)),call.=FALSE)
 })
 cat(name,':',nobs(m),'papers\n');rm(m);gc();bundle
}

complete_sample<-function(d,y,extra=character()) {
 vars<-unique(c(y,main_terms,controls,'reference_member_id','source_id','primary_field_id','year','is_conference','team_size',extra))
 d[complete.cases(d[,..vars])]
}

basis_controls<-c('log_n_sources','reference_source_hhi','mean_ref_age','log_ref_citation_base')
context_index<-function(d,cfg) {
 q<-d[,.(mbin=pmin(floor(mean_pivot*20),19),gbin=pmin(floor(grounding_gap*20),19),team_size,source_count_bin,hhi_bin)]
 totals<-q[,.N,by=.(mbin,gbin)];totals[,cell:=paste(mbin,gbin,sep='_')]
 q<-q[complete.cases(q),.N,by=.(mbin,gbin,team_size,source_count_bin,hhi_bin)]
 q[,`:=`(cell=paste(mbin,gbin,sep='_'),context=paste(team_size,source_count_bin,hhi_bin,sep='_'))]
 list(counts=split(q,by='cell',keep.by=FALSE),totals=setNames(totals$N,totals$cell))
}
context_support<-function(d,grid,ref,cfg,index=NULL) {
 if(is.null(index))index<-context_index(d,cfg)
 cell<-function(m,g)paste(pmin(floor(m*20),19),pmin(floor(g*20),19),sep='_')
 rkey<-cell(ref[1],ref[2]);r<-index$counts[[rkey]];rn<-unname(index$totals[rkey])
 if(is.null(r)||!length(rn)||is.na(rn))return(data.frame(context_cells=rep(0L,nrow(grid)),context_mass_here=0,context_mass_reference=0))
 r<-r[N>=cfg$profile_context_min_count]
 rows<-lapply(seq_len(nrow(grid)),function(i) {
  key<-cell(grid$mean_pivot[i],grid$grounding_gap[i]);q<-index$counts[[key]];qn<-unname(index$totals[key])
  if(is.null(q)||!length(qn)||is.na(qn))return(list(context_cells=0L,context_mass_here=0,context_mass_reference=0))
  q<-q[N>=cfg$profile_context_min_count];common<-intersect(q$context,r$context)
  list(context_cells=length(common),context_mass_here=sum(q[context%in%common,N])/qn,
       context_mass_reference=sum(r[context%in%common,N])/rn)
 })
 as.data.frame(rbindlist(rows))
}
# Optional specifications are explicitly recorded; primary failures stop execution.
fit<-function(e,ctx,d,name,y,...,required=TRUE) {
 reason<-''
 result<-if(required)fit_impl(e,ctx,d,name,y,...,required=TRUE) else
  tryCatch(fit_impl(e,ctx,d,name,y,...,required=FALSE),error=function(err){reason<<-conditionMessage(err);NULL})
 e$model_status[[name]]<-data.table(model=name,outcome=y,n_input=nrow(d),
  status=if(is.null(result))'not_estimated' else 'estimated',
  reason=if(!is.null(result))'' else if(nzchar(reason))reason else 'insufficient complete or estimable papers')
 if(is.null(result))message(name,': ',e$model_status[[name]]$reason)
 result
}
paired_models<-function(e,ctx,d,prefix,y,extra=basis_controls,rich_fe=base_fe,
                         conditional=TRUE,required=TRUE,cluster_ref='reference_member_id') {
 probe<-fit(e,ctx,d,paste0(prefix,'_sample'),y,ctrl=c(controls,extra),fe=rich_fe,
            keep_rows=TRUE,surface=FALSE,required=required,cluster_ref=cluster_ref)
 if(is.null(probe))return(NULL)
 cc<-d[probe$rows]
 a<-fit(e,ctx,cc,paste0(prefix,'_base'),y,conditional_support=conditional,required=required,cluster_ref=cluster_ref)
 b<-fit(e,ctx,cc,paste0(prefix,'_full'),y,ctrl=c(controls,extra),fe=rich_fe,
        conditional_support=conditional,required=required,cluster_ref=cluster_ref)
 if(is.null(a)||is.null(b))return(NULL)
 stopifnot(a$n_estimated==nrow(cc),b$n_estimated==nrow(cc))
 invisible(cc)
}
main_analysis<-function(e,ctx,d) {
 b<-complete_sample(d,'any_novel_pct',basis_controls)
 cc<-paired_models(e,ctx,b,'formation','any_novel_pct')
 # Every control specification below uses exactly this actual estimation sample.
 blocks<-list(count='log_n_sources',concentration='reference_source_hhi',age='mean_ref_age',visibility='log_ref_citation_base',
              count_concentration=c('log_n_sources','reference_source_hhi'))
 for(nm in names(blocks))fit(e,ctx,cc,paste0('formation_add_',nm),'any_novel_pct',
     ctrl=c(controls,blocks[[nm]]),conditional_support=TRUE)
 fit(e,ctx,cc,'formation_mg_support_only','any_novel_pct',conditional_support=FALSE)
 paired_models(e,ctx,b,'formation_venue','any_novel_pct',rich_fe=venue_fe)
 for(y in c('n_reference_sources','source_hhi_pct'))
  fit(e,ctx,b,paste0('reference_profile_',y),y,conditional_support=FALSE)
 if(ctx$cfg$run_nonlinear) {
  flexible<-c('splines::ns(log_n_sources,df=4)','splines::ns(reference_source_hhi,df=4)',
              'splines::ns(mean_ref_age,df=3)','splines::ns(log_ref_citation_base,df=3)')
  paired_models(e,ctx,b,'formation_flexible','any_novel_pct',extra=flexible,required=FALSE)
 }
 if(ctx$cfg$run_structural_fe) {
  structural<-paste(base_fe,'n_reference_sources + hhi_bin + ref_count_bin',sep=' + ')
  paired_models(e,ctx,b,'formation_structural','any_novel_pct',rich_fe=structural,
                extra=c('mean_ref_age','log_ref_citation_base'),required=FALSE)
 }
 if(ctx$cfg$run_splines) {
  f<-features(cc$mean_pivot,cc$grounding_gap,TRUE);z<-copy(cc)
  for(v in names(f))set(z,j=v,value=f[[v]])
  for(suffix in c('base','full'))fit(e,ctx,z,paste0('formation_spline_',suffix),'any_novel_pct',
      terms=names(f),ctrl=c(controls,if(suffix=='full')basis_controls),spline=TRUE,conditional_support=TRUE,required=FALSE)
 }
 invisible(NULL)
}
measure_specificity<-function(e,ctx,d) {
 outcomes<-c('any_novel_pct','novel_share_pct','z10_asinh','z50_asinh','semantic_distance')
 for(y in outcomes)paired_models(e,ctx,complete_sample(d,y,basis_controls),paste0('measure_',y),y,required=FALSE)
 if(!ctx$cfg$run_common_measures)return(invisible(NULL))
 b<-complete_sample(d,outcomes,basis_controls)
 if(nrow(b)<500L){
  e$model_status[['common_measure_sample']]<-data.table(model='common_measure_sample',outcome='multiple',n_input=nrow(b),status='not_estimated',reason='fewer than 500 common-measure papers')
  return(invisible(NULL))
 }
 cc<-paired_models(e,ctx,b,'common_any_novel_pct','any_novel_pct',required=FALSE)
 if(is.null(cc))return(invisible(NULL))
 for(y in setdiff(outcomes,'any_novel_pct'))for(suffix in c('base','full'))
  fit(e,ctx,cc,paste0('common_',y,'_',suffix),y,ctrl=c(controls,if(suffix=='full')basis_controls),
      conditional_support=TRUE,required=FALSE)
 scales<-rbindlist(lapply(outcomes,function(y)data.table(outcome=y,n=nrow(cc),mean=mean(cc[[y]]),sd=sd(cc[[y]]))))
 fwrite(scales,file.path(ctx$out,'tables/common_measure_scales.csv'))
 invisible(NULL)
}
recognition_analysis<-function(e,ctx,d) {
 specs<-list(citation=list(y='citation_percentile',data=d),joint=list(y='joint_success_pct',data=d),
             recognition_novel=list(y='high_recognition_pct',data=d[any_novel==1]))
 for(nm in names(specs)) {
  s<-specs[[nm]];bb<-complete_sample(s$data,s$y,basis_controls)
  paired_models(e,ctx,bb,nm,s$y,required=FALSE)
 }
 for(cut in c(75,95))fit(e,ctx,complete_sample(d[any_novel==1],paste0('high',cut,'_pct'),basis_controls),
       paste0('recognition_cut',cut),paste0('high',cut,'_pct'),ctrl=c(controls,basis_controls),conditional_support=TRUE,required=FALSE)
 if(ctx$cfg$run_ppml)fit(e,ctx,complete_sample(d,'c_total',basis_controls),'citations_PPML','c_total',
                       ctrl=c(controls,basis_controls),surface=FALSE,poisson=TRUE,required=FALSE)
 invisible(NULL)
}
reference_comparisons<-function(e,ctx,d) {
 b<-complete_sample(d,'any_novel_pct',basis_controls)
 # Main model's actual rows are reconstructed by an explicit common-sample fit.
 r<-fit(e,ctx,b,'population_sample','any_novel_pct',ctrl=c(controls,basis_controls),surface=FALSE,keep_rows=TRUE)
 cc<-b[r$rows];pooled<-'primary_field_id^year^is_conference + team_size'
 for(suffix in c('pooled','member'))fit(e,ctx,cc,paste0('population_',suffix),'any_novel_pct',
     ctrl=c(controls,basis_controls),fe=if(suffix=='pooled')pooled else base_fe,conditional_support=TRUE)
 small<-c('log_refs','log_team_h','log_team_output',basis_controls)
 year_fe<-'reference_member_id^year + primary_field_id^year^is_conference + team_size'
 r<-fit(e,ctx,b,'member_year_sample','any_novel_pct',ctrl=small,fe=year_fe,surface=FALSE,keep_rows=TRUE,required=FALSE)
 if(!is.null(r)) {
  cc<-b[r$rows]
  a<-fit(e,ctx,cc,'member_year_matched_member','any_novel_pct',ctrl=small,conditional_support=TRUE,required=FALSE)
  v<-fit(e,ctx,cc,'member_year_matched_year','any_novel_pct',ctrl=small,fe=year_fe,conditional_support=TRUE,required=FALSE)
  if(!is.null(a)&&!is.null(v))stopifnot(a$n_estimated==v$n_estimated)
 }
 invisible(NULL)
}
sensitivity<-function(e,ctx,d) {
 b<-complete_sample(d,'any_novel_pct',basis_controls)
 specs<-list(split0=b[split_half==0],split1=b[split_half==1],exclude2024=b[year<2024],
             exclude2023_2024=b[year<2023],small_teams=b[team_size<=5])
 for(nm in names(specs))paired_models(e,ctx,specs[[nm]],paste0('sensitivity_',nm),'any_novel_pct',required=FALSE)
 sets<-list(diversity=c('team_between_js','team_within_shannon'),collaboration=c('prior_team_edge_density','log_degree'),
       interdisciplinarity='ref_field_rao',status=c('prior_h_gini','prior_output_log_sd'),
       history_depth=c('log_min_history_refs','log_mean_history_refs'))
 for(nm in names(sets)) {
  extra<-sets[[nm]];cc<-complete_sample(b,'any_novel_pct',extra)
  paired_models(e,ctx,cc,paste0('sensitivity_',nm),'any_novel_pct',extra=c(basis_controls,extra),required=FALSE)
 }
 fit(e,ctx,b,'sensitivity_h_reference','any_novel_pct',ctrl=c(controls,basis_controls),
   fe='h_reference_id + primary_field_id^year^is_conference + team_size',cluster_ref='h_reference_id',conditional_support=TRUE,required=FALSE)
 for(alt in c('q25_gap','pivot_sd')) {
  cc<-complete_sample(b,'any_novel_pct',alt);f<-features(cc$mean_pivot,cc[[alt]])
  for(v in names(f))set(cc,j=v,value=f[[v]])
  fit(e,ctx,cc,paste0('alternative_',alt),'any_novel_pct',ctrl=c(controls,basis_controls),surface=FALSE,required=FALSE)
 }
 if(ctx$cfg$run_cohorts)for(yy in seq(1990,2020,5))
  paired_models(e,ctx,b[year>=yy & year<=yy+4],paste0('cohort_',yy),'any_novel_pct',required=FALSE)
 invisible(NULL)
}
field_analysis<-function(e,ctx,d) {
 if(!ctx$cfg$run_fields)return(invisible(NULL))
 b<-complete_sample(d,'any_novel_pct',basis_controls)
 sizes<-b[,.N,by=.(primary_field_id,field_id_text)][order(-N)]
 fwrite(sizes,file.path(ctx$out,'tables/field_sample_sizes.csv'))
 for(i in seq_len(nrow(sizes))) {
  name<-paste0('field_',sizes$primary_field_id[i]);z<-b[primary_field_id==sizes$primary_field_id[i]]
  entry<-data.table(prefix=name,field_code=sizes$primary_field_id[i],field_id=sizes$field_id_text[i],
      n_available=nrow(z),n_estimated=NA_integer_,n_profiles_base=0L,n_profiles_full=0L,
      status='not_estimated',reason='below configured field size')
  if(nrow(z)>=ctx$cfg$field_min_papers) {
   paired_models(e,ctx,z,name,'any_novel_pct',required=FALSE)
   a<-e$results[[paste0(name,'_base')]];v<-e$results[[paste0(name,'_full')]]
   if(!is.null(a)&&!is.null(v)) {
    entry[,`:=`(n_estimated=as.integer(v$n_estimated),n_profiles_base=nrow(a$contrasts),n_profiles_full=nrow(v$contrasts),
        status=if(nrow(v$contrasts))'estimated' else 'estimated_no_profile_support',
        reason=if(nrow(v$contrasts))'' else 'no predefined profile has sufficient common context support')]
   } else entry[,reason:='optional field model not estimated; see model_status.csv']
  }
  e$field_coverage[[name]]<-entry
 }
 invisible(NULL)
}
export_results<-function(e,ctx) {
 b<-e$results
 cf<-rbindlist(lapply(b,`[[`,'coefficients'),use.names=TRUE,fill=TRUE)
 tests<-rbindlist(lapply(b,`[[`,'test'),use.names=TRUE,fill=TRUE)
 if(nrow(tests))tests[,p_BH:=p.adjust(p,'BH')]
 surfaces<-rbindlist(lapply(b,`[[`,'surface'),use.names=TRUE,fill=TRUE)
 pc<-rbindlist(lapply(b,`[[`,'contrasts'),use.names=TRUE,fill=TRUE)
 pc[,`:=`(p_Holm_primary=NA_real_,p_BH_secondary=NA_real_)]
 ix<-which(pc$model%in%c('formation_base','formation_full'))
 if(length(ix))pc[ix,p_Holm_primary:=p.adjust(p,'holm',n=6L)]
 secondary<-which(!pc$model%in%c('formation_base','formation_full'))
 if(length(secondary))pc[secondary,p_BH_secondary:=p.adjust(p,'BH')]
 samples<-rbindlist(lapply(names(b),function(nm)data.table(model=nm,n_input=b[[nm]]$n_input,
  n_complete=b[[nm]]$n_complete,n_estimated=b[[nm]]$n_estimated,
  clusters_reference=b[[nm]]$clusters_reference,clusters_source_year=b[[nm]]$clusters_source_year)))
 fwrite(cf,file.path(ctx$out,'tables/coefficients.csv'));fwrite(tests,file.path(ctx$out,'tables/asymmetry_tests.csv'))
 fwrite(surfaces,file.path(ctx$out,'tables/surfaces.csv'));fwrite(pc,file.path(ctx$out,'tables/profile_contrasts.csv'))
 fwrite(samples,file.path(ctx$out,'tables/model_samples.csv'))
 fwrite(rbindlist(lapply(b,`[[`,'profile_support'),use.names=TRUE,fill=TRUE),file.path(ctx$out,'tables/profile_support.csv'))
 fwrite(rbindlist(e$model_status,use.names=TRUE,fill=TRUE),file.path(ctx$out,'tables/model_status.csv'))
 # Descriptive contrasts between specifications: no cross-model standard error is invented.
 a<-pc[model=='formation_base',.(mean_pivot,gap_from,gap_to,base=estimate)]
 v<-pc[model=='formation_full',.(mean_pivot,gap_from,gap_to,full=estimate)]
 change<-merge(a,v,by=c('mean_pivot','gap_from','gap_to'))
 change[,`:=`(change=full-base,absolute_shrinkage=ifelse(abs(base)>.1,1-abs(full/base),NA_real_))]
 fwrite(change,file.path(ctx$out,'tables/specification_change.csv'))
 scales_path<-file.path(ctx$out,'tables/common_measure_scales.csv')
 if(file.exists(scales_path)) {
  scales<-fread(scales_path);j<-merge(pc[grepl('^common_',model)],scales,by='outcome');j<-j[is.finite(sd)&sd>0]
  for(v in c('estimate','lower','upper','se'))set(j,j=paste0(v,'_sd'),value=j[[v]]/j$sd)
  fwrite(j,file.path(ctx$out,'tables/common_measure_contrasts.csv'))
 }
 if(length(e$field_coverage)) {
  fields<-rbindlist(e$field_coverage,use.names=TRUE,fill=TRUE);fwrite(fields,file.path(ctx$out,'tables/field_coverage.csv'))
  fld<-pc[grepl('^field_',model)];fld[,prefix:=sub('_(base|full)$','',model)]
  fld<-merge(fld,fields,by='prefix');fld[,p_BH_fields:=p.adjust(p,'BH')]
  fwrite(fld,file.path(ctx$out,'tables/field_contrasts.csv'))
 }
 primary<-c('formation_base','formation_full','formation_venue_base','formation_venue_full',
            'citation_full','recognition_novel_full','joint_full')
 table<-cf[model%in%primary];table[,formatted:=sprintf('%.4f (%.4f)',estimate,se)]
 wide<-dcast(table,term~model,value.var='formatted');fwrite(wide,file.path(ctx$out,'tables/main_models.csv'))
 esc<-function(x)gsub('<','&lt;',gsub('&','&amp;',as.character(x),fixed=TRUE),fixed=TRUE)
 rows<-apply(as.matrix(wide),1,function(x)paste0('<tr><td>',paste(esc(x),collapse='</td><td>'),'</td></tr>'))
 writeLines(c('<!doctype html><meta charset="utf-8"><style>body{font:14px sans-serif;margin:30px}td,th{padding:8px;border-bottom:1px solid #ddd}</style>',
  '<h1>Experience configuration and reference-source foundations</h1><p>Associational models. Coefficients with clustered standard errors.</p>',
  paste0('<table><tr><th>',paste(names(wide),collapse='</th><th>'),'</th></tr>'),rows,'</table>'),file.path(ctx$out,'tables/main_models.html'))
 writeLines('ready',file.path(ctx$out,'MODELS_READY'))
}
