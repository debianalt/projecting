# Replication Materials: Multi-dimensional Decomposition of Material Extraction Patterns

## Overview

This repository contains replication materials and supplementary data for the article "Multi-dimensional decomposition of material extraction patterns: A stylised scenario analysis of the MERCOSUR-EU trade agreement."

**Author:** Raimundo Elías Gómez
**Affiliation:** CONICET / Faculty of Humanities and Social Sciences, National University of Misiones (Argentina)
**Contact:** elias.gomez@conicet.gov.ar
**ORCID:** 0000-0002-4468-9618

## Repository Structure

```
projecting/
├── data/                    # Input and intermediate datasets
│   ├── tensor_summary.parquet           # Aggregated extraction by MFA category, country, year
│   ├── material_classification.parquet  # 367 material indicators with MFA classification
│   ├── ntf_loadings.nc                 # NTF component loadings (K=6)
│   ├── ntf_diagnostics.parquet         # Rank selection diagnostics (K=2..10)
│   ├── projections.parquet             # Scenario projections by bloc and MFA category
│   ├── projections_by_sector.parquet   # Sector-level agreement effect estimates
│   └── ne_110m_admin_0_countries.zip   # Natural Earth shapefile for choropleth maps
├── scripts/                 # Analysis scripts (Python)
│   ├── config.py                       # Shared configuration (paths, constants, parameters)
│   ├── 01_build_tensor.py              # Extract MERCOSUR-4 + EU-27 from GLORIA TQ
│   ├── 02_ntf_analysis.py              # Non-negative tensor factorisation (rank selection + K=6)
│   ├── 03_geospatial_viz.py            # Choropleth maps, temporal trends, heatmaps
│   ├── 04_scenario_projection.py       # Baseline/agreement scenario paths + trend bootstrap
│   └── 05_sensitivity_analysis.py      # Shock-magnitude/phase-in sensitivity envelope (Fig 7)
├── figures/                 # All figures (article + supplementary)
├── CODEBOOK.md              # Variable and file definitions
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
└── README.md                # This file
```

## Data Sources

| Dataset | Source | Period | Description |
|---------|--------|--------|-------------|
| Material extraction | GLORIA MRIO (Loop060) satellite accounts | 1990-2022 | Territorial extraction by country, sector, and material |
| Shapefile | Natural Earth | - | 1:110m country boundaries |

The GLORIA MRIO database is available from the Industrial Ecology Virtual Laboratory (https://ielab.info; Lenzen et al., 2017; 2022). The full database (~500 GB) is required only for `01_build_tensor.py`; intermediate outputs for scripts 02-04 are included in `data/`.

## Methodology

### 1. Tensor Construction (01_build_tensor.py)

Extracts material extraction data from GLORIA TQ satellite accounts (physical extension tables recording direct extraction in tonnes by sector, country, and year) for 31 countries (EU-27 + MERCOSUR-4) across 120 sectors, 367 material indicators, and 33 years (1990-2022). Material indicators are classified into 15 subcategories following EUROSTAT/UNEP-IRP conventions. Requires the full GLORIA database.

### 2. Non-negative Tensor Factorisation (02_ntf_analysis.py)

Decomposes the 4D tensor (31 countries x 120 sectors x 15 material subcategories x 33 years) using Non-negative PARAFAC. Ranks K=2..10 are screened under identical settings (three initialisations each); variance explained rises smoothly with no sharp elbow, so K=6 is adopted as a parsimonious, interpretable rank (R-squared=0.75) rather than a statistically unique optimum. The final K=6 model is refitted with 10 initialisations. Outputs component loadings for each dimension.

### 3. Geospatial Visualisation (03_geospatial_viz.py)

Generates choropleth maps of country loadings, temporal evolution by bloc, material subcategory heatmaps, and historical extraction trends by MFA category.

### 4. Scenario Paths (04_scenario_projection.py)

Constructs baseline (no-agreement) and agreement paths for the temporal loadings (2022-2034) using quadratic trend extrapolation with illustrative, tariff-schedule-informed shocks (December 2024 factsheets). The shocks are scenario assumptions set by the analyst, not estimated trade-policy responses; the magnitude of any effect is by construction proportional to the assumed shock. Augmented Dickey-Fuller tests confirm trend-stationarity of all six temporal loading series (p < 0.05 for five components; p = 0.079 for C6 under constant+trend, p = 0.017 under constant-only). A residual bootstrap (N=1000, 90%) characterises dispersion around the fitted historical trend only — not the agreement-effect uncertainty. Agreement assumed to enter into force in 2027 with a 5-year linear phase-in.

### 5. Sensitivity Analysis (05_sensitivity_analysis.py)

Sweeps the full shock vector over scale factors 0.5-1.5 and the phase-in window over 3-10 years (20 combinations) to map the sensitivity envelope of the MFA-category effect (Fig 7). The ordering biomass > metal ores > non-metallic minerals is preserved in all 20 combinations, whereas the magnitudes scale with the assumed shock; an equal-shock alternative confirms that the cross-category differentiation is induced by the assumed concession ordering. Runs from the intermediate data alone (the full GLORIA tensor is not needed).

## Reproduction Instructions

### Requirements

```bash
pip install -r requirements.txt
```

Python 3.10+ required.

### Execution Order

Scripts must be run in order from the `scripts/` directory:

```bash
cd scripts

# 1. Build tensor from GLORIA (requires full GLORIA database; ~15 min)
python 01_build_tensor.py

# 2. NTF decomposition (~10 min)
python 02_ntf_analysis.py

# 3. Geospatial visualisations (~2 min)
python 03_geospatial_viz.py

# 4. Scenario paths (~5 min)
python 04_scenario_projection.py

# 5. Sensitivity envelope (~1 min; uses ntf_loadings.nc only)
python 05_sensitivity_analysis.py
```

**Partial replication (without GLORIA):** Scripts 02-04 can be run using the intermediate data files included in `data/`. Only `01_build_tensor.py` requires the full GLORIA database. To replicate from step 02 onwards, ensure `data/tensor_summary.parquet` and `data/material_classification.parquet` are present, then run scripts 02-04 in order.

**Note:** `01_build_tensor.py` produces `data/tensor_materials.nc` (360 MB), which is excluded from this repository. This file is consumed by `02_ntf_analysis.py`; its outputs (`ntf_loadings.nc`, `ntf_diagnostics.parquet`) are included, enabling replication from step 02 without the full tensor.

### Expected Outputs

- `data/ntf_loadings.nc`: NTF component loadings
- `data/ntf_diagnostics.parquet`: Rank selection diagnostics
- `data/projections.parquet`: Scenario projections
- `data/projections_by_sector.parquet`: Sector-level estimates
- `figures/fig*.png`: All figures

## Sample (N = 31)

**EU-27:** Austria, Belgium, Bulgaria, Croatia, Cyprus, Czech Republic, Denmark, Estonia, Finland, France, Germany, Greece, Hungary, Ireland, Italy, Latvia, Lithuania, Luxembourg, Malta, Netherlands, Poland, Portugal, Romania, Slovakia, Slovenia, Spain, Sweden

**MERCOSUR-4:** Argentina, Brazil, Paraguay, Uruguay

## Figure Mapping

| Article Figure | File | Description |
|----------------|------|-------------|
| Fig. 1 | `fig01_pipeline.png` | Analytical pipeline flowchart |
| Fig. 2 | `fig02_historical_trends.png` | Material extraction by MFA category (1990–2022) |
| Fig. 3 | `fig03_material_heatmap.png` | Material subcategory loadings on NTF components |
| Fig. 4 | `fig04_choropleth_loadings.png` | Choropleth maps of country loadings (6 components) |
| Fig. 5 | `fig05_projection_scenarios.png` | Scenario paths: baseline vs agreement (trend-only 90% bootstrap bands) |
| Fig. 6 | `fig06_material_impact.png` | Central-scenario change by MFA category |
| Fig. 7 | `fig07_sensitivity.png` | Sensitivity envelope of the MFA-category effect to assumed shocks |
| Fig. S1 | `fig_s1_component_loadings.png` | Full component loadings (all dimensions) |
| Fig. S2 | `fig_s2_sector_comparison.png` | Sector loadings weighted by bloc |
| Fig. S3 | `fig_s3_rank_selection.png` | NTF rank selection: R-squared and marginal gain |
| Fig. S4 | `fig_s4_temporal_blocs.png` | Temporal evolution of components by bloc |

## Key Results

1. **Structural asymmetry:** MERCOSUR-4 extracts approximately 6,100 megatonnes annually, comparable to the EU-27 (5,500 Mt), but concentrated in biomass and metal ores rather than construction minerals.

2. **Six latent components:** NTF identifies agricultural crops, forestry, fishery, non-metallic minerals, livestock and grazing, and metal ores as distinct extraction patterns (K=6, R-squared=0.75).

3. **Scenario sensitivity:** Under the central illustrative scenario, the largest sector effects fall on cattle raising (+10.8%) and cereal cultivation (+9.7%), with biomass strongest at the MFA-category level (+8.5%). These magnitudes are conditional on the assumed shocks; across all 20 sensitivity combinations the robust feature is the ordering biomass (+3.0% to +12.7%) > metal ores (+2.1% to +8.8%) > non-metallic minerals (+0.7% to +3.0%), not the point values.

## Reproducibility

All random seeds are fixed via `RANDOM_SEED = 42` in `config.py`. Results are fully reproducible given the same GLORIA database version (Loop060).

## References

Lenzen, M., Geschke, A., Abd Rahman, M. D., et al. (2017). The Global MRIO Lab. *Economic Systems Research*, 29(2), 158-186. https://doi.org/10.1080/09535314.2017.1301887

Lenzen, M., Geschke, A., West, J., et al. (2022). Implementing the material footprint to measure progress towards Sustainable Development Goals 8 and 12. *Nature Sustainability*, 5(2), 157-166. https://doi.org/10.1038/s41893-021-00811-6

## License

Code: MIT. Data: subject to GLORIA MRIO terms of use.

## Citation

If you use these materials, please cite:

```
Gómez, R. E. (2026). Multi-dimensional decomposition of material extraction
patterns: A stylised scenario analysis of the MERCOSUR-EU trade agreement.
```

**Zenodo concept DOI (latest version):** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18674715.svg)](https://doi.org/10.5281/zenodo.18674715)
