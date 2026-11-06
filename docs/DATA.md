# Input data

The workflow reads a SciSciNet2-derived research directory and a prepared graph directory configured in the project settings.

## Research tables

The default research directory is data/SciSciNet2_research. Configure another location with data_root in config.json.

| Relative path | Key fields and contents |
| --- | --- |
| core/papers.parquet | paper_id, year, source_id, primary_field_id, is_conference, reference counts, citation indicators, team size, and prior team summaries |
| core/authors_year.parquet | author_id, year, prior_papers, prior_h, career_age |
| metrics/author_paper_pivot.parquet | paper_id, author_id, year, reference_source_pivot, new_reference_source_share, prior_window_reference_count, current_reference_count, pivot_observed |
| metrics/source_novelty.parquet | paper_id, new_source_pair_share, source_pair_z10, source_pair_z50, source_pair_z_coverage, n_reference_sources, reference_source_hhi |
| metrics/team_diversity.parquet | Optional precomputed historical team profiles and diversity indicators |
| metrics/collaboration_network.parquet | Optional precomputed collaboration-network indicators |
| metrics/interdisciplinarity.parquet | Optional precomputed reference-field indicators |
| metrics/embedding_field_history.parquet | Optional paper_id, emb_field_history_count, emb_field_history_distance |

The required paper columns and optional measurement columns are defined by CORE, AUTHOR, PIVOT, and MODULES in data.py. Each optional module selects its analyses from the supplied measurements.

Identifiers must remain exact integers or exact character identifiers. Do not convert paper or author identifiers through floating point.

## Prepared graph tables

The default graph directory is data/SciSciNet2_prepared. Configure raw_root and the two filenames in benchmark_config.json.

| File | Required columns |
| --- | --- |
| sciscinet_authors_paperid.parquet | paperid, authorid |
| sciscinet_paperrefs.parquet | citing_paperid, cited_paperid |

The graph inputs reconstruct complete source histories for author-history reassignment.

## Existing annual analysis data

A completed notebook 01 directory contains the DATA_READY marker and annual data/*.parquet files. Set analysis_run_dir to this directory when using an existing panel.

The default analysis period is 1990-2024. The member-source history uses the preceding three publication years. Precomputed semantic distance uses the historical field representation described in MEASURES.md.
