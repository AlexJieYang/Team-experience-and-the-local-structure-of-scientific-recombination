# Measurement definitions

## Research core and temporal ordering

The research core contains papers with a unique known journal or conference source after exclusion of nonresearch document types and paratext. A retained reference edge has both endpoints in the core. Paper pairs are deduplicated globally, self-citations by identical paper ID are removed, and future references are excluded. References to papers in the same publication year are allowed.

The analysis includes focal publication years through 2024. Historical source pairs and author histories use records from strictly earlier focal years.

## Member-source distance

For paper p, let x_ps count references to source s. For author a in focal year y, let h_ay sum the full source-count vectors of the author's core papers from years y-3 through y-1. Each historical paper contributes its full vector to every listed author.

$$
d_{ap}=1-\frac{x_p^\top h_{ay}}{\|x_p\|\|h_{ay}\|}.
$$

The distance lies in [0, 1] for nonnegative source-count vectors and is missing if either norm is zero. The precomputed reference_source_pivot field stores this distance. Historical-reference count and current-reference count are the sums of h_ay and x_p, respectively.

For a team with n listed members:

$$
M_p=\frac{1}{n}\sum_a d_{ap},\qquad G_p=M_p-\min_a d_{ap}.
$$

History matching does not truncate the selected authors' histories.

## New source-pair share

Let n_ps be the number of reference slots assigned to source s. Unordered slot-pair multiplicities are:

$$
m_{p,st}=n_{ps}n_{pt}\quad(s<t),\qquad
m_{p,ss}=n_{ps}(n_{ps}-1)/2.
$$

Within-source pairs are retained. The denominator is the number of unordered reference-slot pairs, not the number of distinct source-pair categories.

Let I_st,<y indicate whether source pair (s, t) occurred in a core paper strictly before focal year y. Papers in the same focal year use the same prior-year history.

$$
\mathrm{share}_p=
\frac{\sum_{s\le t}m_{p,st}\mathbf{1}[I_{st,<y}=0]}{\binom{N_p}{2}}.
$$

The share is missing for fewer than two valid source-assigned references. For reference sources (A, A, B, C), the pair multiplicities are AA:1, AB:2, AC:2, and BC:1. If AB is the only new pair, the share is 2/6.

Formation is 100 times the indicator that the share is positive. Share regressions use 100 times the share. Source HHI is the sum of squared source proportions; source count is the number of categories with positive reference counts.

## Source-pair conventionality

The precomputed source-slot benchmark reassigns source labels within citing-year by cited-year strata. Its documented defaults are 10 reassignments and seed 20261013. It preserves each paper's reference count, reference-age strata, and within-stratum source margins.

A pair z score is its observed count minus the randomized mean, divided by the randomized sample standard deviation. A zero randomized standard deviation produces a missing score.

The paper-level z10 and z50 indicators use integer reference-slot-pair multiplicities and linear weighted quantiles at 0.1 and 0.5. They do not weight distinct pair categories equally. Models use asinh-transformed scores; common-sample displays use the relevant outcome standard deviation.

The source-slot benchmark defines the input conventionality indicators. It is separate from the author-history reassignment implemented by notebooks 04-06.

## Semantic distance from field history

A valid paper embedding e_p has finite entries and a positive norm. Let u_p be its unit-normalized vector. For the focal primary field f and year y, sum valid unit vectors of same-field papers from years y-5 through y-1:

$$
H_{fy}=\sum_{r\in\mathcal{H}_{fy}}u_r,\qquad
D_p=1-\frac{u_p^\top H_{fy}}{\|H_{fy}\|}.
$$

At least 20 valid historical vectors are required. The distance is missing for an invalid focal embedding, a missing field, insufficient history, or a zero historical-center norm. It lies in [0, 2]. Same-year papers enter the historical representation only after all papers in their year are scored.

The five-year interval defines a backward knowledge history rather than a future outcome window. The measure is not nearest-neighbor distance and is not constructed from reference sources. Encoder and embedding-generation details belong to the dataset's construction metadata.

## Executable definitions

metric_definitions.py contains small executable definitions of these measurements. Production analyses use the precomputed outcomes. The history benchmark reconstructs source profiles and member-to-paper distances to implement the conditional reference comparison.
