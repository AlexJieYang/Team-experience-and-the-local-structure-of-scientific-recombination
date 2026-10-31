# Assemble completed model worlds and shared-support summaries.
suppressPackageStartupMessages({library(data.table);library(jsonlite)})

singleton_keep_rows <- function(d) {
  if(!nrow(d))return(integer())
  combine <- paste(d$primary_field_id,d$year,d$is_conference,sep="\r")
  groups <- lapply(list(d$reference_member_id,combine,d$team_size),function(x)match(x,unique(x)))
  keep <- seq_len(nrow(d))
  repeat {
    remove <- rep(FALSE,length(keep))
    for(g in groups) {
      counts <- tabulate(g[keep],nbins=max(g))
      remove <- remove|counts[g[keep]]==1L
    }
    if(!any(remove)||!length(keep))break
    keep <- keep[!remove]
  }
  keep
}

completed_sample <- function(panel,ys,baseline,reference,expected_hash,
                           saved_ids=NULL,fixed_ids=NULL,allow_sample_probe=FALSE) {
  required <- unique(c("paper_id","reference_member_id","primary_field_id","source_id","year",
      "is_conference","team_size",baseline,reference,ys,"mean_pivot","grounding_gap"))
  complete <- copy(panel[complete.cases(panel[,..required])])
  choose <- if(!is.null(fixed_ids))fixed_ids else saved_ids
  if(!is.null(choose)) {
    if(anyNA(choose)||anyDuplicated(choose))stop("Fixed estimation IDs must be unique")
    ix <- match(choose,complete$paper_id)
    if(anyNA(ix))stop("A saved estimation paper is outside its original complete panel")
    d <- complete[ix]
  } else d <- complete[singleton_keep_rows(complete)]
  if(!identical(fingerprint(d$paper_id),expected_hash)) {
    if(!isTRUE(allow_sample_probe)||!is.null(choose))
      stop("The completed estimation sample differs from the saved sample identity. ",
           "If no saved IDs exist, allow_sample_probe=TRUE permits one observed sample-selection fit only.")
    d <- sample_rows(panel,ys,baseline,reference)$data
    if(!identical(fingerprint(d$paper_id),expected_hash))stop("Observed sample-selection fit does not match the saved sample")
  }
  if(anyDuplicated(d$paper_id))stop("The completed sample contains duplicated papers")
  list(data=d,complete=complete[,.(paper_id,year)])
}

shared_support_from_cache <- function(out,design,checkpoints,B,legacy=FALSE) {
  target <- file.path(out,"model_tables",paste0(design,"_shared_support_cells.csv"))
  paths <- file.path(checkpoints,sprintf("%s_world_%04d_support.rds",design,0:B))
  ready <- file.path(checkpoints,sprintf("%s_world_%04d.ready",design,0:B))
  if(file.exists(target)&&all(file.exists(paths))&&
     file.info(target)$mtime>=max(file.info(ready)$mtime))return("available")
  shared <- data.table(mb=integer(),gb=integer(),weight=numeric(),mean_pivot=numeric(),g_from=numeric(),g_to=numeric())
  status <- "unavailable_missing_support"
  if(!legacy&&all(file.exists(paths))) {
    state <- readRDS(paths[1]);eligible <- state$eligible;minimum_weights <- state$weight
    for(path in paths[-1]) {
      z <- readRDS(path)
      if(!identical(z$mb,state$mb)||!identical(z$gb,state$gb))stop("Support grid differs across worlds")
      eligible <- eligible&z$eligible;minimum_weights <- pmin(minimum_weights,z$weight)
    }
    take <- which(eligible);shared <- state[take,.(mb,gb)]
    shared[,weight:=minimum_weights[take]]
    if(nrow(shared))shared[,`:=`(weight=weight/sum(weight),mean_pivot=(mb+.5)/20,g_from=(gb+.5)/20,g_to=(gb+1.5)/20)]
    else shared <- data.table(mb=integer(),gb=integer(),weight=numeric(),mean_pivot=numeric(),g_from=numeric(),g_to=numeric())
    status <- "available"
  }
  fwrite(shared,target);status
}

collect_completed_scenario <- function(project,scenario,bridge_ids=NULL,allow_sample_probe=FALSE) {
  bc <- active_settings(project);out <- file.path(bc[["_out"]],"null",scenario)
  settings_path <- file.path(out,"scenario_settings.json")
  if(!file.exists(settings_path))stop("Missing scenario settings: ",scenario)
  settings <- fromJSON(settings_path);B <- as.integer(settings$replicates)
  common <- c("any_novel_pct","novel_share_pct","z10_asinh","z50_asinh","semantic_distance")
  designs <- list(main="any_novel_pct")
  if(isTRUE(settings$run_common_measures))designs$common <- common
  if(!is.null(bridge_ids)) {
    if(any(bridge_ids[["design"]]=="main"))designs$bridge_main <- "any_novel_pct"
    if(any(bridge_ids[["design"]]=="common"))designs$bridge_common <- common
  }
  if(file.exists(file.path(out,"SCENARIO_EMPTY")))return(data.table(scenario=scenario,design=names(designs),
    status="empty",worlds_complete=0L,worlds_expected=B+1L,n=NA_integer_,reason="No matched teams"))
  tables <- file.path(out,"model_tables");dir.create(tables,recursive=TRUE,showWarnings=FALSE)
  all_ids_path <- file.path(tables,"estimation_paper_ids.csv")
  all_ids <- if(file.exists(all_ids_path))fread(all_ids_path,colClasses=c(paper_id="character")) else NULL
  panel <- NULL;panel_md5 <- NULL;rows <- list();sizes <- list();years <- list();statuses <- list();good <- character()
  baseline <- c("log_refs","log_team_h","log_team_output","log_ref_output","log_ref_h")
  reference <- c("log_n_sources","reference_source_hhi","mean_ref_age","log_ref_citation_base")
  for(design in names(designs)) {
    ys <- designs[[design]];is_bridge <- startsWith(design,"bridge_")
    checkpoints <- design_checkpoint_dir(out,design)
    ready <- file.path(checkpoints,sprintf("%s_world_%04d.ready",design,0:B))
    complete_files <- file.exists(ready)
    for(suffix in c("stats","profiles","coefficients"))
      complete_files <- complete_files&file.exists(file.path(checkpoints,sprintf(paste0("%s_world_%04d_",suffix,".csv"),design,0:B)))
    status <- data.table(scenario=scenario,design=design,status="pending",worlds_complete=sum(complete_files),
      worlds_expected=B+1L,n=NA_integer_,local_support_status="unavailable",reason="Not all expected worlds are complete")
    if(!all(complete_files)) {
      old <- file.path(out,"model_checkpoints",sprintf("%s_world_%04d.ready",design,0:B))
      if(is_bridge&&any(file.exists(old))) {
        status[,`:=`(status="needs_correct_sample",reason="This design requires estimates on its own fixed paper sample")]
      }
      statuses[[design]] <- status;next
    }
    signature_path <- file.path(checkpoints,paste0(design,"_settings.json"))
    if(!file.exists(signature_path))stop("Completed design has no sample/settings signature: ",scenario," / ",design)
    signature <- fromJSON(signature_path)
    if(!identical(as.character(signature$outcomes),as.character(ys)))stop("Saved outcomes differ: ",scenario," / ",design)
    if(is.null(panel)) {
      panel <- read_panel(file.path(out,"panel.parquet"))
      if(anyDuplicated(panel$paper_id))stop("Repeated matched paper")
      panel_md5 <- unname(tools::md5sum(file.path(out,"panel.parquet")))
    }
    if(!identical(panel_md5,signature$panel))stop("Panel file differs from the one used for estimation: ",scenario)
    if(!identical(as.character(signature$controls),baseline)||!identical(as.character(signature$reference),reference)||
       !identical(signature$FE,base_fe)||!identical(signature$cluster,"reference_member_id + source_id^year"))
      stop("Saved model specification differs: ",scenario," / ",design)
    saved_ids <- NULL;ids_path <- file.path(checkpoints,paste0(design,"_paper_ids.csv"))
    if(file.exists(ids_path))saved_ids <- fread(ids_path,colClasses=c(paper_id="character"))$paper_id
    else if(!is.null(all_ids)&&any(all_ids[["design"]]==design))saved_ids <- paper_ids_for_design(all_ids,design)
    fixed_ids <- if(is_bridge)paper_ids_for_design(bridge_ids,sub("bridge_","",design)) else NULL
    selected <- completed_sample(panel,ys,baseline,reference,signature$sample,saved_ids,fixed_ids,allow_sample_probe)
    d <- selected$data;n <- nrow(d)
    for(suffix in c("stats","profiles","coefficients")) {
      paths <- file.path(checkpoints,sprintf(paste0("%s_world_%04d_",suffix,".csv"),design,0:B))
      q <- rbindlist(lapply(paths,fread),fill=TRUE)
      if(!all(q[["design"]]==design)||!setequal(unique(q$world),0:B))stop("Unexpected design/world labels in saved tables")
      if(suffix=="stats") {
        key <- q[, .N,by=.(world,outcome,spec)]
        if(any(key$N!=1L)||nrow(key)!=(B+1L)*length(ys)*2L||any(q$n!=n))stop("Saved model rows or sample size differ")
        if(!all(q$outcome%in%ys)||!all(q$spec%in%c("baseline","conditioned"))||any(!is.finite(q$partial_r2_gap)))
          stop("Invalid saved global statistics")
      }
      if(suffix=="profiles"&&any(q$n!=n))stop("Saved profile sample size differs")
      fwrite(q,file.path(tables,paste0(design,"_assembled_",suffix,".csv")))
    }
    rows[[design]] <- data.table(design=design,paper_id=d$paper_id)
    sizes[[design]] <- data.table(design=design,matched=if(is_bridge)n else nrow(panel),
      complete=if(is_bridge)n else nrow(selected$complete),estimated=n,
      clusters_member=uniqueN(d$reference_member_id),clusters_source_year=uniqueN(d[,.(source_id,year)]))
    years[[design]] <- rbindlist(list((if(is_bridge)d else panel)[,.(papers=.N),by=year][,stage:="matched"],
      (if(is_bridge)d else selected$complete)[,.(papers=.N),by=year][,stage:="complete"],
      d[,.(papers=.N),by=year][,stage:="estimated"]),fill=TRUE)[,design:=design]
    scales <- data.table(outcome=ys,n=n,sd=sapply(ys,function(y)sd(d[[y]])))
    if(any(!is.finite(scales$sd))||any(scales$sd<=0))stop("Invalid outcome scales")
    fwrite(scales,file.path(tables,paste0(design,"_outcome_scales.csv")))
    fwrite(d[,.(paper_id)],ids_path)
    support_status <- shared_support_from_cache(out,design,checkpoints,B,isTRUE(settings$legacy_extension))
    set(status,j="status",value="available");set(status,j="n",value=n)
    set(status,j="local_support_status",value=support_status);set(status,j="reason",value="")
    statuses[[design]] <- status;good <- c(good,design)
    message(scenario," / ",design,": assembled ",B+1L," completed worlds; N = ",format(n,big.mark=","),"; using saved estimates")
  }
  available <- rbindlist(statuses,fill=TRUE)
  if("main"%in%good) {
    for(suffix in c("stats","profiles","coefficients")) {
      paths <- file.path(tables,paste0(good,"_assembled_",suffix,".csv"))
      fwrite(rbindlist(lapply(paths,fread),fill=TRUE),file.path(tables,paste0("all_",suffix,".csv")))
      unlink(paths)
    }
    fwrite(rbindlist(rows),all_ids_path)
    fwrite(rbindlist(sizes),file.path(tables,"sample_sizes.csv"))
    fwrite(rbindlist(years),file.path(tables,"sample_sizes_by_year.csv"))
    fwrite(available[,.(design,status,reason,n)],file.path(tables,"design_status.csv"))
    writeLines("ready",file.path(out,"BENCHMARK_MODELS_READY"))
  }
  available
}

collect_completed_benchmarks <- function(project=getwd(),allow_sample_probe=FALSE) {
  project <- normalizePath(project,mustWork=TRUE)
  source(file.path(project,"benchmark_models.R"),local=globalenv())
  source(file.path(project,"analysis.R"),local=globalenv())
  bc <- active_settings(project);setDTthreads(bc$n_threads_R)
  path <- file.path(bc[["_out"]],"scenario_plan.json")
  if(!file.exists(path))stop("Missing scenario plan; complete notebook 04 or select an existing benchmark directory")
  plan <- fromJSON(path,simplifyVector=FALSE)
  source_plan <- Filter(function(x)identical(x$role,"source"),plan)
  ordered <- c(source_plan,Filter(function(x)!identical(x$role,"source"),plan))
  availability <- list();bridge <- NULL
  for(item in ordered) {
    scenario <- item$scenario
    z <- collect_completed_scenario(project,scenario,if(identical(item$role,"primary"))bridge else NULL,allow_sample_probe)
    availability[[scenario]] <- z
    if(identical(item$role,"source")&&any(z$design=="main"&z$status=="available")) {
      bridge <- fread(file.path(bc[["_out"]],"null",scenario,"model_tables/estimation_paper_ids.csv"),colClasses=c(paper_id="character"))
    }
  }
  result <- rbindlist(availability,fill=TRUE)
  fwrite(result,file.path(bc[["_out"]],"model_availability.csv"))
  print(result)
  if(any(result$status%in%c("pending","needs_correct_sample")))
    message("Completed estimates are ready for notebook 06. Incomplete designs remain listed as pending.")
  invisible(result)
}
