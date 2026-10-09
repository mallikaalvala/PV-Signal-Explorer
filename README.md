# PV Signal Explorer V2

A Streamlit research application for exploratory pharmacovigilance signal detection in FAERS/AEMS-style spontaneous-report data. It combines pooled disproportionality analysis with subgroup stratification, cross-stratum heterogeneity, multiplicity control, masking/reversal review, clinical-context modules, graphical exploration, and exportable reports.

> **Important:** Results are hypothesis-generating. Disproportionality does not establish causality, incidence, or absolute risk. Use the software for research/educational signal review and validate data preparation and case-level findings before publication or regulatory interpretation.

## Features

- PRR, ROR, 95% ROR CI, Pearson chi-square and Yates sensitivity statistic.
- Age, sex, reporter-type and report-source stratification.
- Cochran Q, I² and Benjamini-Hochberg FDR-adjusted heterogeneity tests.
- Exploratory masked-signal and direction-reversal detection.
- Seriousness/death summaries from OUTC.
- Exploratory indication-confounding and time-to-onset modules.
- Data-quality/completeness dashboard.
- Signal Evidence Card and multidimensional robustness profile.
- Forest plots plus reusable visualization helpers for top signals, signal maps, subgroup heatmaps and completeness charts.
- CSV and self-contained HTML report exports.

## Repository layout

```text
pv_signal_explorer_v2/
├── app.py
├── src/pv_signal_explorer/
│   ├── __init__.py
│   ├── core.py
│   └── visualizations.py
├── data/
│   ├── README.md
│   └── sample/
│       ├── DEMO.csv
│       ├── DRUG.csv
│       └── REAC.csv
├── tests/test_core.py
├── docs/METHODOLOGY.md
├── requirements.txt
├── pyproject.toml
├── .streamlit/config.toml
├── .gitignore
└── LICENSE
```

## Installation

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
streamlit run app.py
```

## Minimum input

Upload matching DRUG, REAC and DEMO files. CSV and raw `$`-delimited FAERS ASCII formats are supported. OUTC, INDI, THER and RPSR are optional.

The sample files are synthetic and are intended only to verify the UI. They are too small for meaningful signal interpretation.

## Statistical interpretation

The default screening rule uses minimum pair count + PRR ≥ 2 + Pearson χ² ≥ 4. ROR and its 95% CI are displayed as complementary evidence. Subgroup RORs are estimated independently within each stratum. Cross-stratum inconsistency is summarized using Cochran Q and I²; BH-FDR is supplied because many drug-event pairs may be tested.

A masked-signal flag is an exploratory prioritization rule, not a causal claim. A direction reversal is displayed separately from ordinary heterogeneity.

## Known limitations / development priorities

1. Implement explicit CASEID/CASEVERSION latest-version deduplication before publication-grade use.
2. Preserve DRUG_SEQ linkage for drug-specific THER/INDI temporal and indication analyses.
3. Normalize verbatim products to active ingredients/active moieties.
4. Add validated MedDRA hierarchy handling where licensed/appropriate.
5. Add longitudinal quarter-level persistence analysis and validated Bayesian signal methods only after unit/reference testing.
6. Validate against known positive and negative control drug-event pairs.

## Testing

```bash
pytest -q
```

## Reproducibility

Record the FAERS/AEMS release/quarters, extraction date, input checksums, software version, filters, thresholds, and all data-cleaning decisions for every analysis intended for a manuscript.

## License

MIT. Regulatory source data and MedDRA terminology may have separate terms/licensing requirements and are not distributed with this repository.

## Bayesian and temporal surveillance (V3 analytics)
The repository includes an open two-component Gamma-Poisson empirical-Bayes model that reports **EBGM, EB05 and EB95**, plus a **BCPNN-style Information Component (IC, IC025, IC975)**. A quarterly surveillance engine recomputes each period's reporting background and displays signal emergence, persistence, quarter-over-quarter change, and EBGM/IC trajectories.


