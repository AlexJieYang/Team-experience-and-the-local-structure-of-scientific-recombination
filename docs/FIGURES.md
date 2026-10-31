# Figures

## Export settings

Publication plates use a full width of 178 mm, a 7-point base font, and 8-point panel titles by default. Panel aspect ratios can be set in figure_settings.json.

PDF and SVG exports retain vector geometry. Final PNG files use 600 dpi; notebook previews use 150 dpi. Generated captions and numerical plotting data are saved alongside the figure outputs.

## Observational figures

| Figure | Panels |
| --- | --- |
| Fig1 | Two by three: configuration examples, sample density, formation, source count, HHI, and publication cohorts |
| Fig2 | Two by three: fixed-profile contrasts, reference controls, common contexts, two model surfaces, and functional forms |
| Fig3 | Two by three: five outcomes, conditioned contrast matrix, contrast plane, source count, HHI, and sample comparison |
| Fig4 | Three-column layout: fields, cohorts, recognition, specification series, and recognition among formation papers |
| S1 | Ridgeline and raincloud distributions; positive shares and zero mass are distinguished |
| S2 | Cross-sectional alluvial composition, cohorts, and within-profile composition |
| S3 | Five by three common-sample atlas: baseline, conditioned, and local conditioned slopes |
| S4 | Three by three uncertainty and support display |
| S5 | Specification contrasts and component matrix |
| S6 | Supported field-specific contrast intervals |
| S7 | Two by three publication-cohort heatmaps |
| S8 | Two by two static 3D surfaces and contours |
| S9 | Correlations, hexagonal count plots, and configuration relationships |
| S10 | Sample coverage, the full field-by-target domain, and endpoint/common-context counts |

Paired baseline and conditioned maps share color scales. The common-sample atlas uses each outcome's actual estimation-sample standard deviation within a row. Local-slope panels have their own scales.

Alluvial ribbons represent cross-sectional counts. Publication cohorts group years and do not define future outcome windows. Unsupported cells remain masked rather than being replaced with zero.

## History benchmark figures

| Figure | Panels and conditions |
| --- | --- |
| B1 | Global gap-block reference distributions, common-sample outcomes, supported local contrasts, and common-support coordinates |
| B2 | Two by three: observed/random configuration density, excess mass, M/G cumulative distributions, and overlap |
| B3 | Two by three: sample retention, annual actual estimation counts, donor-stratum sizes, own-history retention, team-history retention, and source-profile similarity |
| B4 | Three panels comparing matching methods on identical actual estimation papers; requires completed source-matched and broad bridge designs |
| B5 | Optional comparison of completed paper-hash and member-cluster benchmarks, including actual sample sizes and Monte Carlo ranks |

B1 uses gray ranges for central 95% randomized reference intervals and separate colored intervals for observed cluster-based 95% intervals. When local and fixed-profile contrasts are unavailable, the plate contains only the supported upper row.

B2 uses shared logarithmic density scales and a symmetric-log difference scale. Marginal cumulative distributions and fixed-grid overlap describe the occupied configuration domain.

B3 uses the actual fixed-effect estimation samples. Historical-source similarity uses the first 20 randomized worlds and at most 3,000 queried author-years per year; it does not use outcome values.

B4 checks identical paper IDs and observed statistics before comparing matching schemes. B5 retains different sample and world counts explicitly.

Fixed-profile estimates with unavailable support remain identifiable in detailed tables. The supported_fixed_profiles.csv export contains only comparisons satisfying the stated support rule.
