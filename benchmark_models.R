suppressPackageStartupMessages({
  library(fixest);library(data.table);library(arrow);library(jsonlite)
})

read_panel <- function(path) {
  d <- as.data.table(read_parquet(path))
  for(v in intersect(c("paper_id","reference_member_id","source_id","primary_field_id"),names(d)))
    set(d,j=v,value=as.character(d[[v]]))
  for(v in names(d)) {
    if(inherits(d[[v]],"integer64"))set(d,j=v,value=as.numeric(d[[v]]))
    if(is.numeric(d[[v]]))set(d,which(!is.finite(d[[v]])),v,NA_real_)
  }
  d
}

active_settings <- function(project) {
  p <- file.path(project,"active_benchmark.json")
  bc <- fromJSON(if(file.exists(p))p else file.path(project,"benchmark_config.json"))
  if(is.null(bc[["_out"]]))bc[["_out"]] <- file.path(project,"outputs",bc$run_name)
  bc
}

fingerprint <- function(x) {
  path <- tempfile(fileext=".json");on.exit(unlink(path))
  writeLines(toJSON(x,auto_unbox=TRUE,null="null",digits=NA),path)
  unname(tools::md5sum(path))
}

paper_ids_for_design <- function(ids,target_design) {
  if(!all(c("design","paper_id")%in%names(ids)))stop("Missing design/paper_id columns")
  take <- which(as.character(ids[["design"]])==target_design)
  selected_ids <- as.character(ids[["paper_id"]][take])
  if(anyNA(selected_ids)||any(!nzchar(selected_ids))||anyDuplicated(selected_ids))
    stop("Missing or duplicated paper IDs within design: ",target_design)
  selected_ids
}

design_checkpoint_dir <- function(out,design) {
  root <- file.path(out,"model_checkpoints")
  # Store paired-sample designs separately from full-panel designs.
  if(startsWith(design,"bridge_"))file.path(root,"exact_sample") else root
}


fit_many <- function(d,ys,ctrl,gap=TRUE,remove="none") {
  lhs <- if(length(ys)==1L)ys else paste0("c(",paste(ys,collapse=","),")")
  terms <- c("m","m2",if(gap)c("g","g2","mg"),ctrl)
  f <- as.formula(paste(lhs,"~",paste(terms,collapse="+"),"|",
      "reference_member_id + primary_field_id^year^is_conference + team_size"))
  result <- feols(f,data=d,vcov=~reference_member_id+source_id^year,
    fixef.rm=remove,notes=FALSE,warn=TRUE,mem.clean=TRUE)
  if(length(ys)==1L)setNames(list(result),ys) else
    setNames(lapply(seq_along(ys),function(i)result[[i]]),ys)
}

fixed_support <- function(d,cfg) {
  idx <- context_index(d,cfg)
  rbindlist(lapply(c(.4,.6,.8),function(m) {
    pts <- data.frame(mean_pivot=c(m,m),grounding_gap=c(.05,.25))
    counts <- support_counts(d,pts)
    mb <- pmin(floor(d$mean_pivot*20),19L);gb <- pmin(floor(d$grounding_gap*20),19L)
    a <- d[mb==min(floor(m*20),19L)&gb==1L,.N,by=team_size][N>=20L]
    b <- d[mb==min(floor(m*20),19L)&gb==5L,.N,by=team_size][N>=20L]
    cc <- context_support(d,pts[2,,drop=FALSE],c(m,.05),cfg,index=idx)
    size_ok <- length(intersect(a$team_size,b$team_size))>0L
    density_ok <- all(counts>=cfg$surface_cell_min)
    context_ok <- cc$context_cells[1]>0L&&cc$context_mass_here[1]>=cfg$profile_common_mass_min&&cc$context_mass_reference[1]>=cfg$profile_common_mass_min
    data.table(mean_pivot=m,n_from=counts[1],n_to=counts[2],common_size=size_ok,
      context_mass_from=cc$context_mass_reference[1],context_mass_to=cc$context_mass_here[1],
      supported=density_ok&&size_ok&&context_ok,
      reason=if(!density_ok)"endpoint_density" else if(!size_ok)"common_team_size" else if(!context_ok)"reference_context" else "")
  }))
}

local_support <- function(d,cfg) {
  # All 0.05-wide adjacent gap-bin comparisons are specified in advance.
  q <- d[,.(mb=pmin(floor(mean_pivot*20),19L),gb=pmin(floor(grounding_gap*20),19L),
            team_size,source_count_bin,hhi_bin)]
  total <- q[,.N,by=.(mb,gb)];setnames(total,"N","n_from")
  upper <- copy(total);upper[,gb:=gb-1L];setnames(upper,"n_from","n_to")
  edges <- merge(CJ(mb=0:19,gb=0:18),total,by=c("mb","gb"),all.x=TRUE)
  edges <- merge(edges,upper,by=c("mb","gb"),all.x=TRUE)
  contexts <- q[complete.cases(q),.N,by=.(mb,gb,team_size,source_count_bin,hhi_bin)]
  contexts <- contexts[N>=cfg$profile_context_min_count]
  other <- copy(contexts);other[,gb:=gb-1L];setnames(other,"N","N_to")
  shared <- merge(contexts,other,by=c("mb","gb","team_size","source_count_bin","hhi_bin"))
  mass <- shared[,.(shared_from=sum(N),shared_to=sum(N_to)),by=.(mb,gb)]
  edges <- merge(edges,mass,by=c("mb","gb"),all.x=TRUE,sort=TRUE)
  for(v in c("n_from","n_to","shared_from","shared_to"))set(edges,which(is.na(edges[[v]])),v,0)
  edges[,`:=`(mass_from=fifelse(n_from>0,shared_from/n_from,0),mass_to=fifelse(n_to>0,shared_to/n_to,0))]
  edges[,eligible:=n_from>=cfg$surface_cell_min & n_to>=cfg$surface_cell_min &
          mass_from>=cfg$profile_common_mass_min & mass_to>=cfg$profile_common_mass_min]
  edges[,weight:=pmin(n_from,n_to)]
  edges[,.(mb,gb,eligible,weight)]
}

profile_estimates <- function(model,support) {
  b <- coef(model);V <- vcov(model);needed <- c("g","g2","mg")
  if(!all(needed%in%names(b)))stop("Gap block is rank deficient")
  rbindlist(lapply(seq_len(nrow(support)),function(i) {
    L <- setNames(rep(0,length(b)),names(b))
    L["g"] <- .2;L["g2"] <- -.02;L["mg"] <- .2*(support$mean_pivot[i]-.5)
    estimate <- sum(L*b);se <- sqrt(max(0,as.numeric(t(L)%*%V%*%L)))
    cbind(support[i],data.table(estimate=estimate,se=se,lower=estimate-1.96*se,upper=estimate+1.96*se))
  }))
}

sample_rows <- function(panel,ys,baseline,reference,legacy_ids=NULL) {
  ids <- c("paper_id","reference_member_id","primary_field_id","source_id","year","is_conference","team_size")
  required <- unique(c(ids,baseline,reference,ys,"mean_pivot","grounding_gap"))
  d <- copy(panel[complete.cases(panel[,..required])]);complete <- copy(d[,.(paper_id,year)])
  if(!is.null(legacy_ids)) {
    if(anyNA(legacy_ids)||anyDuplicated(legacy_ids))stop("Fixed estimation IDs must be unique and nonmissing")
    ix <- match(legacy_ids,d$paper_id)
    if(anyNA(ix))stop("An existing estimation paper is missing from its fixed panel")
    d <- d[ix]
  } else if(nrow(d)>=500L) {
    f <- features(d$mean_pivot,d$grounding_gap)
    for(v in names(f))set(d,j=v,value=f[[v]])
    probe <- fit_many(d,ys[1],c(baseline,reference),remove="perfect_fit")[[1]]
    d <- d[fixest::obs(probe)];rm(probe);gc(verbose=FALSE)
  }
  list(data=d,complete=complete)
}

run_benchmark <- function(project=getwd(),scenario=NULL,bridge_ids=NULL) {
  project <- normalizePath(project,mustWork=TRUE)
  source(file.path(project,"analysis.R"),local=globalenv())
  cfg <- fromJSON(file.path(project,"config.json"));bc <- active_settings(project)
  if(!is.null(scenario))bc$scenario <- scenario
  out <- file.path(bc[["_out"]],"null",bc$scenario)
  if(file.exists(file.path(out,"SCENARIO_EMPTY")))return(invisible(FALSE))
  settings <- fromJSON(file.path(out,"scenario_settings.json"))
  for(k in names(settings))bc[[k]] <- settings[[k]]
  if(!file.exists(file.path(out,"PERMUTATIONS_READY")))stop("Run notebook 04 first")
  cfg$surface_cell_min <- bc$profile_surface_cell_min
  setFixest_nthreads(bc$n_threads_R);setDTthreads(bc$n_threads_R)
  panel <- read_panel(file.path(out,"panel.parquet"))
  if(anyDuplicated(panel$paper_id))stop("Repeated matched paper")
  baseline <- c("log_refs","log_team_h","log_team_output","log_ref_output","log_ref_h")
  reference <- c("log_n_sources","reference_source_hhi","mean_ref_age","log_ref_citation_base")
  common <- c("any_novel_pct","novel_share_pct","z10_asinh","z50_asinh","semantic_distance")
  designs <- list(main="any_novel_pct")
  if(isTRUE(bc$run_common_measures))designs$common <- common
  if(!is.null(bridge_ids)&&nrow(bridge_ids)) {
    if(any(bridge_ids$design=="main"))designs$bridge_main <- "any_novel_pct"
    if(any(bridge_ids$design=="common"))designs$bridge_common <- common
  }
  tables <- file.path(out,"model_tables");dir.create(tables,recursive=TRUE,showWarnings=FALSE)
  checkpoints <- file.path(out,"model_checkpoints");dir.create(checkpoints,showWarnings=FALSE)
  legacy <- isTRUE(bc$legacy_extension)
  old_ids <- if(legacy&&file.exists(file.path(tables,"estimation_paper_ids.csv")))
    fread(file.path(tables,"estimation_paper_ids.csv"),colClasses=c(paper_id="character")) else NULL
  rows <- list();sizes <- list();year_sizes <- list();statuses <- list()
  for(design in names(designs)) {
    ys <- designs[[design]];fixed_ids <- NULL
    is_bridge <- startsWith(design,"bridge_")
    if(is_bridge)fixed_ids <- paper_ids_for_design(bridge_ids,sub("bridge_","",design))
    if(legacy&&!is.null(old_ids))fixed_ids <- paper_ids_for_design(old_ids,design)
    checkpoints <- design_checkpoint_dir(out,design)
    dir.create(checkpoints,recursive=TRUE,showWarnings=FALSE)
    saved_ids_path <- file.path(checkpoints,paste0(design,"_paper_ids.csv"))
    if(is.null(fixed_ids)&&file.exists(saved_ids_path))
      fixed_ids <- fread(saved_ids_path,colClasses=c(paper_id="character"))$paper_id
    selected <- sample_rows(panel,ys,baseline,reference,fixed_ids);d <- selected$data
    if(anyDuplicated(d$paper_id))stop("An estimation sample contains duplicate papers")
    if(nrow(d)<500L) {
      statuses[[design]] <- data.table(design=design,status="not_estimated",reason="fewer than 500 actual estimation papers")
      next
    }
    rows[[design]] <- data.table(design=design,paper_id=d$paper_id)
    statuses[[design]] <- data.table(design=design,status="estimated",reason="",n=nrow(d))
    sizes[[design]] <- data.table(design=design,matched=if(is_bridge)length(fixed_ids) else nrow(panel),
      complete=if(is_bridge)length(fixed_ids) else nrow(selected$complete),estimated=nrow(d),
      clusters_member=uniqueN(d$reference_member_id),clusters_source_year=uniqueN(d[,.(source_id,year)]))
    year_sizes[[design]] <- rbindlist(list(
      (if(is_bridge)d else panel)[,.(papers=.N),by=year][,stage:="matched"],
      (if(is_bridge)d else selected$complete)[,.(papers=.N),by=year][,stage:="complete"],
      d[,.(papers=.N),by=year][,stage:="estimated"]),fill=TRUE)[,design:=design]
    sd_y <- sapply(ys,function(y)sd(d[[y]]))
    if(any(!is.finite(sd_y))||any(sd_y<=0))stop("Outcome SD must be positive")
    fwrite(data.table(outcome=ys,n=nrow(d),sd=sd_y),file.path(tables,paste0(design,"_outcome_scales.csv")))
    invariant <- settings[setdiff(names(settings),c("replicates","run_common_measures","role","legacy_extension"))]
    signature <- list(version=2L,outcomes=ys,controls=baseline,reference=reference,FE=base_fe,
      cluster="reference_member_id + source_id^year",sample=fingerprint(d$paper_id),
      panel=unname(tools::md5sum(file.path(out,"panel.parquet"))),matching=invariant,
      support=list(density=cfg$surface_cell_min,context_n=cfg$profile_context_min_count,context_mass=cfg$profile_common_mass_min),
      fixest=as.character(packageVersion("fixest")))
    path <- file.path(checkpoints,paste0(design,"_settings.json"))
    if(file.exists(path)&&!identical(fromJSON(path,simplifyVector=FALSE),fromJSON(toJSON(signature,auto_unbox=TRUE,null="null",digits=NA),simplifyVector=FALSE)))
      stop("Model settings or estimation papers changed; choose a new scenario to keep checkpoints separate")
    if(!file.exists(path)&&!legacy&&file.exists(file.path(checkpoints,sprintf("%s_world_0000.ready",design))))
      stop("Existing model checkpoints have no compatible settings")
    write_json(signature,path,auto_unbox=TRUE,null="null",pretty=TRUE,digits=NA)
    fwrite(d[,.(paper_id)],file.path(checkpoints,paste0(design,"_paper_ids.csv")))
    for(world in 0:bc$replicates) {
      prefix <- file.path(checkpoints,sprintf("%s_world_%04d",design,world))
      if(file.exists(paste0(prefix,".ready")))next
      z <- copy(d)
      if(world>0) {
        random <- read_panel(file.path(out,"replicates",sprintf("replicate_%04d.parquet",world)))
        ix <- match(d$paper_id,random$paper_id)
        if(anyNA(ix)||anyDuplicated(random$paper_id))stop("Randomized paper IDs differ")
        set(z,j="mean_pivot",value=random$mean_pivot[ix]);set(z,j="grounding_gap",value=random$grounding_gap[ix])
      }
      if(any(!is.finite(z$mean_pivot))||any(!is.finite(z$grounding_gap)))stop("Invalid coordinates")
      f <- features(z$mean_pivot,z$grounding_gap)
      for(v in names(f))set(z,j=v,value=f[[v]])
      support <- fixed_support(z,cfg)
      if(!legacy)saveRDS(local_support(z,cfg),paste0(prefix,"_support.rds"))
      stats <- list();profiles <- list();coefs <- list()
      for(spec in c("baseline","conditioned")) {
        ctrl <- c(baseline,if(spec=="conditioned")reference)
        restricted <- fit_many(z,ys,ctrl,FALSE);full <- fit_many(z,ys,ctrl,TRUE)
        for(y in ys) {
          model <- full[[y]];small <- restricted[[y]]
          stopifnot(nobs(model)==nrow(z),nobs(small)==nrow(z))
          rss0 <- sum(resid(small)^2);rss1 <- sum(resid(model)^2)
          if(!is.finite(rss0)||rss0<=0)stop("Restricted residual variance is not positive")
          key <- paste(spec,y,sep="_");V <- vcov(model);b <- coef(model)
          if(!all(c("g","g2","mg")%in%names(b)))stop("Gap block is rank deficient")
          stats[[key]] <- data.table(world=world,design=design,outcome=y,spec=spec,n=nrow(z),
            partial_r2_gap=max(0,min(1,(rss0-rss1)/rss0)),rss_restricted=rss0,rss_full=rss1,
            clusters_member=uniqueN(z$reference_member_id),clusters_source_year=uniqueN(z[,.(source_id,year)]),
            b_g=b[["g"]],b_g2=b[["g2"]],b_mg=b[["mg"]],v_gg=V["g","g"],v_gg2=V["g","g2"],
            v_gmg=V["g","mg"],v_g2g2=V["g2","g2"],v_g2mg=V["g2","mg"],v_mgmg=V["mg","mg"])
          q <- profile_estimates(model,support)
          for(v in c("world","design","outcome","spec","n"))
            set(q,j=v,value=switch(v,world=world,design=design,outcome=y,spec=spec,n=nrow(z)))
          q[,`:=`(estimate_sd=estimate/sd_y[[y]],lower_sd=lower/sd_y[[y]],upper_sd=upper/sd_y[[y]])]
          profiles[[key]] <- q
          cf <- as.data.table(coeftable(model),keep.rownames="term")
          setnames(cf,c("term","estimate","se","statistic","p"))
          for(v in c("world","design","outcome","spec"))set(cf,j=v,value=switch(v,world=world,design=design,outcome=y,spec=spec))
          coefs[[key]] <- cf
        }
        rm(restricted,full);gc(verbose=FALSE)
      }
      fwrite(rbindlist(stats,fill=TRUE),paste0(prefix,"_stats.csv"));fwrite(rbindlist(profiles,fill=TRUE),paste0(prefix,"_profiles.csv"))
      fwrite(rbindlist(coefs,fill=TRUE),paste0(prefix,"_coefficients.csv"));writeLines("ready",paste0(prefix,".ready"))
      message(bc$scenario," / ",design," / world ",world," of ",bc$replicates)
    }
    support_paths <- file.path(checkpoints,sprintf("%s_world_%04d_support.rds",design,0:bc$replicates))
    shared <- data.table(mb=integer(),gb=integer(),weight=numeric(),mean_pivot=numeric(),g_from=numeric(),g_to=numeric())
    if(!legacy&&all(file.exists(support_paths))) {
      state <- readRDS(support_paths[1]);eligible <- state$eligible;minimum_weights <- state$weight
      for(p in support_paths[-1]) {
        s <- readRDS(p);stopifnot(identical(s$mb,state$mb),identical(s$gb,state$gb))
        eligible <- eligible&s$eligible;minimum_weights <- pmin(minimum_weights,s$weight)
      }
      take <- which(eligible);shared <- state[take,.(mb,gb)];shared[,weight:=minimum_weights[take]]
      if(nrow(shared))shared[,`:=`(weight=weight/sum(weight),mean_pivot=(mb+.5)/20,g_from=(gb+.5)/20,g_to=(gb+1.5)/20)]
    }
    fwrite(shared,file.path(tables,paste0(design,"_shared_support_cells.csv")))
  }
  fwrite(rbindlist(statuses,fill=TRUE),file.path(tables,"design_status.csv"))
  if(!length(rows)) {
    writeLines("No estimable design",file.path(out,"BENCHMARK_NOT_ESTIMABLE"));return(invisible(FALSE))
  }
  fwrite(rbindlist(rows),file.path(tables,"estimation_paper_ids.csv"))
  fwrite(rbindlist(sizes),file.path(tables,"sample_sizes.csv"));fwrite(rbindlist(year_sizes),file.path(tables,"sample_sizes_by_year.csv"))
  for(suffix in c("stats","profiles","coefficients")) {
    paths <- unlist(lapply(names(rows),function(design)file.path(design_checkpoint_dir(out,design),sprintf(paste0("%s_world_%04d_",suffix,".csv"),design,0:bc$replicates))))
    if(!all(file.exists(paths)))stop("A model world is incomplete")
    fwrite(rbindlist(lapply(paths,fread),fill=TRUE),file.path(tables,paste0("all_",suffix,".csv")))
  }
  writeLines("ready",file.path(out,"BENCHMARK_MODELS_READY"))
  availability_path <- file.path(bc[["_out"]],"model_availability.csv")
  fresh <- data.table(scenario=bc$scenario,design=names(rows),status="available",
    worlds_complete=bc$replicates+1L,worlds_expected=bc$replicates+1L,
    n=vapply(rows,nrow,integer(1)),local_support_status=if(legacy)"unavailable_missing_support" else "available",reason="")
  if(file.exists(availability_path)) {
    previous <- fread(availability_path)
    previous <- previous[which(previous[["scenario"]]!=bc$scenario)]
    fresh <- rbindlist(list(previous,fresh),fill=TRUE)
  }
  fwrite(fresh,availability_path);invisible(TRUE)
}

run_all_benchmarks <- function(project=getwd(),only_scenarios=NULL,include_bridges=TRUE) {
  bc <- active_settings(project);path <- file.path(bc[["_out"]],"scenario_plan.json")
  if(!file.exists(path))return(run_benchmark(project,bc$scenario))
  plan <- fromJSON(path,simplifyVector=FALSE)
  source_plan <- Filter(function(x)identical(x$role,"source"),plan)
  bridge <- NULL
  if(isTRUE(include_bridges)&&length(source_plan)) {
    name <- source_plan[[1]]$scenario
    ok <- run_benchmark(project,name)
    ids <- file.path(bc[["_out"]],"null",name,"model_tables/estimation_paper_ids.csv")
    if(isTRUE(ok)&&file.exists(ids))bridge <- fread(ids,colClasses=c(paper_id="character"))
  }
  for(s in plan) {
    if(identical(s$role,"source")&&isTRUE(include_bridges))next
    if(!is.null(only_scenarios)&&!s$scenario%in%only_scenarios)next
    run_benchmark(project,s$scenario,if(identical(s$role,"primary"))bridge else NULL)
  }
  invisible(NULL)
}
