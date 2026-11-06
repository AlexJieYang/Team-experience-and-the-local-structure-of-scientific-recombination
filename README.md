# Team experience and scientific recombination

This repository contains the Python and R workflow for studying how team members' prior experience relates to the knowledge combinations in a scientific paper. The analysis uses SciSciNet2-derived data and precomputed indicators for publication years 1990-2024.

Python handles data preparation, sparse history reconstruction, and figures. R handles model estimation with fixest. Jupyter notebooks provide the analysis entry points.

## Repository contents

| File | Purpose |
| --- | --- |
| [01_Data.ipynb](01_Data.ipynb) | Prepare the annual analysis panel and descriptive statistics |
| [02_Models_R.ipynb](02_Models_R.ipynb) | Estimate observational models in R |
| [03_Figures.ipynb](03_Figures.ipynb) | Render observational main and supplementary figures |
| [04_History_Benchmark_Python.ipynb](04_History_Benchmark_Python.ipynb) | Reconstruct and reassign complete prior author histories |
| [05_Benchmark_Models_R.ipynb](05_Benchmark_Models_R.ipynb) | Estimate benchmark models or assemble completed worlds |
| [05_Completed_Results_R.ipynb](05_Completed_Results_R.ipynb) | Optional entry point for completed benchmark results |
| [06_Benchmark_Figures_Python.ipynb](06_Benchmark_Figures_Python.ipynb) | Summarize benchmark comparisons and render figures |
| Python modules | Data processing, matching, figures, and result exports |
| R scripts | Observational models, benchmark models, and completed-result summaries |
| [config.json](config.json) | Observational sample and resource settings |
| [benchmark_config.json](benchmark_config.json) | History benchmark, matching, and assignment settings |
| [figure_settings.json](figure_settings.json) | Figure dimensions, fonts, and export resolution |
| [docs](docs/) | Data requirements, measure definitions, design, and figure guide |

The notebooks and supporting modules remain together at the repository root so their imports work without installing the project as a package.

## Installation

Install the Python dependencies from the repository root:

    python -m pip install -r requirements.txt

Install R separately, then run the following in an R console:

    source("install_R.R")

The R installer installs IRkernel, fixest, arrow, data.table, jsonlite, and tidyselect, and registers the R kernel for Jupyter.

Start Jupyter from the repository root:

    jupyter lab

Use the Python kernel for notebooks 01, 03, 04, and 06. Use the R kernel for notebooks 02 and 05.

## Data configuration

The underlying data and generated analysis directories are not included. Supply the SciSciNet2-derived research tables and prepared graph tables described in [docs/DATA.md](docs/DATA.md).

Set these paths before running the workflow:

| Configuration field | Meaning |
| --- | --- |
| config.json: data_root | Research directory containing core/ and metrics/ |
| config.json: run_name | Name of the observational analysis directory |
| benchmark_config.json: raw_root | Prepared author-paper and reference-edge directory |
| benchmark_config.json: analysis_run_dir | Completed notebook 01 output directory; null selects the configured local observational run |
| benchmark_config.json: run_name | Name of the benchmark directory |
| benchmark_config.json: previous_run_dir | Optional paper-hash benchmark directory for the sampling comparison or an existing run to extend |

The example paths are relative to the repository root. Absolute paths to data stored elsewhere can be used instead. Keep the source data consistent with the configured observation cutoff of 2024.

The active_benchmark.json file is generated locally when notebook 04 selects a run. It is excluded from version control.

## Running the analysis

For an analysis starting from the input tables, run notebooks 01 through 06 in numerical order. The default estimate mode in notebook 05 estimates missing worlds and uses compatible completed worlds when available.

| Available inputs | Entry point |
| --- | --- |
| Source tables only | 01, 02, 03, 04, 05, 06 |
| Completed annual panel and observational models | 04, 05, 06 |
| Completed observational model exports | 03 |
| Completed benchmark models and support files | 05_Completed_Results_R, then 06 |
| Benchmark tables already assembled | 06 |

The optional completed-results notebook does not estimate additional regressions.

For another completed results directory, set RESULTS_ROOT in notebook 03. For an existing benchmark directory, select its run_name in benchmark_config.json. If an active_benchmark.json file is present, it selects the active benchmark run.

## Analysis design

The team configuration uses mean member-to-paper source distance M and the mean-minus-minimum gap G. Outcomes include new source-pair formation, new-pair share, source-pair conventionality, and semantic distance from a historical field center.

The default history benchmark selects 35% of indexing-member clusters from the eligible annual panel and uses 999 randomized worlds in addition to the observed world. Each reassignment preserves focal references, outcomes, team slots, covariates, and fixed-effect labels.

| Scenario | History matching and assignment |
| --- | --- |
| broad | Publication year, dominant historical field, and historical-reference-count bin |
| source_matched | Broad matching plus dominant historical cited source |
| source_no_team | Source matching with restricted single-cycle assignments excluding the focal member and observed teammates |
| narrow_history | A narrower historical-reference-count bin |

The broad conditioned formation statistic is the primary global comparison. Secondary local contrasts use the same supported cell pairs and weights in every world. Restricted single-cycle assignments are structural reference comparisons rather than unrestricted permutation tests.

See [docs/RESEARCH_DESIGN.md](docs/RESEARCH_DESIGN.md) and [docs/MEASURES.md](docs/MEASURES.md) for the specifications and measurement definitions.

## Output directories

Observational results are written to:

    outputs/<observational-run-name>/
        data/
        tables/
        pnas_figures/

Benchmark results are written to:

    outputs/<benchmark-run-name>/
        null/<scenario>/
            model_checkpoints/
            model_tables/
            figures/
            manuscript/
        comparison_figures/

Outputs include coefficient tables, actual estimation samples, global and local reference comparisons, shared-support coordinates, captions, plotting data, and manuscript-ready text or LaTeX summaries.

PDF and SVG files retain vector geometry. Final PNG files use 600 dpi by default; notebook previews use 150 dpi. Figure layouts and interpretation are described in [docs/FIGURES.md](docs/FIGURES.md).

## Notebook sources

Notebooks contain source cells without saved outputs, execution counts or execution metadata. Run the appropriate stages with the required external inputs to generate numerical results and figures. Completed result directories can be reused through the entry points listed above.

## Resources and continuing a run

The default process limit is 88 GiB, with at least 12 GiB of available system memory. Data are processed in batches using pandas, Arrow I/O, and SciPy sparse matrices. Python and R are run in separate kernels.

Completed stages, annual histories, assignment worlds, and model worlds are reused when their settings and samples match. Increasing replicates can append worlds within the same assignment design. Use a separate run_name for a different sample or input dataset and a separate scenario for different matching rules.

## Static source checks

```bash
python tools/check_sources.py
```

GitHub Actions runs the same checks for Python syntax, JSON and notebook structure, including empty outputs and execution metadata. It does not execute notebooks, scientific analyses, randomization, model estimation or tests. R cells are checked as notebook source text; R syntax is not validated by this workflow.

`.gitignore` excludes local datasets, generated result directories, runtime files and local active-run settings. Keep the notebook files and supporting modules together at the repository root.

## Citation and license

[CITATION.cff](CITATION.cff) provides the code citation metadata. The research code is available under the [MIT License](LICENSE). SciSciNet2-derived data, graph tables and other external inputs retain their source licenses and access conditions.
