# Research design

## Question and configuration

We study how the distribution of prior experience within a research team relates to the knowledge that its paper assembles. Mean member-to-paper distance M records the team's average position relative to the focal source base. The mean-minus-minimum gap G records the closest member's position at that same average.

This configuration complements individual pivot measures and separates familiarity with source components from novelty of source combinations. The gap is related mechanically to team size, mean distance, and the feasible distance bounds, so models include team-size effects and comparisons are restricted to empirical support.

## Sample and outcomes

The eligible panel covers publication years 1990-2024 and uses the precomputed measurements and eligibility rules in config.json. The history benchmark samples 35% of indexing-member clusters by deterministic hash, retaining every eligible paper of a selected indexing member within that panel.

The primary outcome is new source-pair occurrence in percentage points. Additional outcomes are new-pair share, asinh(z10), asinh(z50), and semantic distance from historical field knowledge. The five-outcome comparison uses an identical actual estimation sample.

Higher z scores indicate more conventional source combinations. Greater semantic distance records greater departure from the historical field center. Citation and recognition analyses are secondary and depend on the supplied observation horizon.

## Observational models

Models include centered M, G, their squares, and their interaction. Baseline controls include reference count, team prior output and h index, and indexing-member prior output and h index. Reference-conditioned models add source count, source HHI, mean reference age, and the prior citation base of cited work.

Fixed effects are indexing member, primary field by publication year by conference status, and exact team size. Standard errors use indexing-member and source-by-year clustering. Baseline and conditioned comparisons within a pair use the same actual estimation papers.

## Conditional author-history reference

Each annual donor pool contains the focal authors plus a deterministic 5% sample of other active authors, capped at 350,000 additional donors per year. Histories require at least 20 valid reference slots and career ages of 1-70 years.

The dominant historical field is the most frequent primary field of the author's three-year history, with identifier ordering resolving ties. Historical-reference-count bins use floor(log2(H)/0.5).

The broad scenario permutes complete histories uniformly within year, dominant historical field, and history-length strata. Fixed points are allowed. Source matching additionally holds the dominant historically cited source constant. The minimum donor pool is 30 in the broad scenario and 10 in the source-matched scenario.

The same author-year mapping is used across all focal papers. A paper is retained only when every listed member is eligible. Every randomized world recalculates source-based member distances, M, and G while keeping the focal reference structure and other regression inputs fixed.

The source-matched actual estimation papers also define broad bridge_main and bridge_common comparisons. These paired comparisons require identical papers in identical order across matching designs.

Restricted source_no_team assignments exclude a focal author and observed focal teammates using rejection sampling from uniform single cycles. An outcome-independent pilot uses 200 candidates and requires at least three feasible cycles. This is a structural reference distribution rather than a uniform distribution over all feasible bijections. The narrow_history scenario uses log2-history bins of width 0.25.

## Global and local statistics

The primary global statistic is the reference-conditioned gap-block partial R-squared in broad/main. It compares the full model with a restricted model excluding g, g squared, and m times g.

Monte Carlo tail ranks use one plus the number of extreme randomized worlds, divided by B+1. The default is 999 randomized worlds plus the observed world. Formal permutation interpretations require exchangeability within the specified strata.

The planned secondary global Holm family contains 37 comparisons under the default design. The conditioned secondary local family contains 19. Planned but unavailable comparisons retain their places in these families. Restricted single-cycle comparisons are excluded from formal permutation families.

Local comparisons use adjacent G-bin centers at the same M-bin center on a fixed 20-by-20 grid. Endpoints each require at least 100 papers. Shared exact-team-size by source-count-bin by HHI-bin contexts each require at least 20 papers at both endpoints and at least 10% context mass at both endpoints.

Eligible cell pairs are intersected across the observed world and every randomized world. Weights are the minimum endpoint counts across all worlds, normalized to sum to one. Every world uses the same coordinates and weights. The contrast increases G by 0.05. No shared domain yields an unavailable contrast.

## Interpretation

Global statistics summarise explanatory strength. Local contrasts describe directional associations on shared coordinates. Observed cluster intervals and randomized reference ranges provide their respective uncertainty summaries.

The configuration-overlap coefficient on the fixed grid is the sum of minimum observed and randomized probability masses. It describes distributional proximity; local comparisons use the support criteria defined above.

Paired matching comparisons use identical estimation papers. Paper-hash and member-cluster sampling comparisons use different samples and are reported as sampling sensitivity. Outcome definitions and supported domains remain explicit in both comparisons.
