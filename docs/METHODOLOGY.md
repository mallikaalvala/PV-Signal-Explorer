# Methodology and reporting notes

## Analysis unit
The application constructs case-level drug-event co-occurrence tables. Counts should be interpreted only after appropriate source-data deduplication and normalization.

## Disproportionality
For a 2×2 table with cells a, b, c and d, the app reports PRR and ROR, a Wald 95% confidence interval for ROR, Pearson chi-square, and a Yates-corrected sensitivity statistic. Zero-cell correction is used for heterogeneity log-OR estimation where needed.

## Subgroup analysis
The denominator universe is reconstructed within each subgroup rather than filtering an already-computed pooled statistic. Supported dimensions are age band, sex, reporter type and report source.

## Heterogeneity
Cochran Q is computed on inverse-variance-weighted log RORs. I² summarizes the proportion of observed variability attributable to cross-stratum heterogeneity. Benjamini-Hochberg FDR-adjusted p-values are reported for large-scale screening.

## Masking/reversal terminology
A masked-signal flag requires an aggregate-negative pair, at least one subgroup-positive result, and statistically supported heterogeneity under the selected threshold. Direction reversal is reported separately and is the closer analogue of a Simpson-type reversal.

## Interpretation
Spontaneous-report disproportionality is hypothesis-generating. It does not estimate incidence or prove that a product caused an event. Results require clinical review and assessment of duplicates, follow-up versions, confounding, missingness, stimulated reporting, channeling, product naming, terminology mapping, and individual case narratives.

## Empirical Bayes (EBGM)
PV Signal Explorer fits a two-component Gamma prior to latent reporting ratios and a Poisson likelihood for observed counts given the expected count under independence. Hyperparameters are estimated by marginal maximum likelihood. The posterior mixture yields EBGM = exp(E[log(lambda)|data]) and 5th/95th posterior percentiles EB05/EB95. This is an open MGPS-style implementation, not a reproduction of proprietary Empirica Signal.

## Information Component (IC)
The IC module uses a BCPNN-style log2 observed-to-expected formulation with Jeffreys-style smoothing and an approximate uncertainty interval. IC025 > 0 is exposed as a screening reference. It is not claimed to reproduce UMC's production implementation exactly.

## Temporal surveillance
For each calendar quarter, drug-event observed and expected counts are reconstructed using that quarter as its own reporting universe. EBGM/EB05 and IC/IC025 are followed longitudinally. The interface reports first alert period, number of alert periods, longest/current persistence run, log2-EBGM trend and quarter-over-quarter change. These are surveillance descriptors, not incidence or causal estimates.
