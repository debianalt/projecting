# Replication Materials: Projecting Material Extraction under the MERCOSUR-EU Trade Agreement

## Overview

This repository contains replication materials and supplementary data for the article "Projecting material extraction under the MERCOSUR-EU trade agreement."

**Author:** Raimundo Elias Gomez
**Affiliations:** CONICET / National University of Misiones (Argentina); Institute of Sociology, University of Porto (Portugal)
**Contact:** rgomez@letras.up.pt
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
│   └── 04_scenario_projection.py       # Trade agreement scenario projections + bootstrap CI
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

Extracts material extraction satellite accounts from GLORIA TQ for 31 countries (EU-27 + MERCOSUR-4) across 120 sectors, 367 material indicators, and 33 years (1990-2022). Material indicators are classified into 15 subcategories following EUROSTAT/UNEP-IRP conventions. Requires the full GLORIA database.

### 2. Non-negative Tensor Factorisation (02_ntf_analysis.py)

Decomposes the 4D tensor (31 countries x 120 sectors x 15 material subcategories x 33 years) using Non-negative PARAFAC with 10 random initialisations per rank. Rank selection based on R-squared and marginal gain identifies K=6 as optimal (R-squared=0.75). Outputs component loadings for each dimension.

### 3. Geospatial Visualisation (03_geospatial_viz.py)

Generates choropleth maps of country loadings, temporal evolution by bloc, material subcategory heatmaps, and historical extraction trends by MFA category.

### 4. Scenario Projections (04_scenario_projection.py)

Projects material extraction under baseline and agreement scenarios (2022-2034) using trend extrapolation with trade agreement shocks calibrated from published tariff schedules (December 2024). Bootstrap confidence intervals (N=1000 replications, 90% CI). Agreement assumed to enter into force in 2027 with a 5-year linear phase-in.

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

# 4. Scenario projections (~5 min)
python 04_scenario_projection.py
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
| Figure 1 | `fig01_pipeline.png` | Analytical pipeline flowchart |
| Figure 2 | `fig07_historical_trends.png` | Material extraction by MFA category (1990-2022) |
| Figure 3 | `fig01_rank_selection.png` | NTF rank selection: R-squared and marginal gain |
| Figure 4 | `fig05_material_heatmap.png` | Material subcategory loadings on NTF components |
| Figure 5 | `fig03_choropleth_loadings.png` | Choropleth maps of country loadings (6 components) |
| Figure 6 | `fig04_temporal_blocs.png` | Temporal evolution of components by bloc |
| Figure 7 | `fig08_projection_scenarios.png` | Projected temporal loadings: baseline vs agreement (90% CI) |
| Figure 8 | `fig09_material_impact.png` | Projected change by MFA category and bloc |
| **Fig. S1** | `fig02_component_loadings.png` | **Supplementary:** Full component loadings (all dimensions) |
| **Fig. S2** | `fig06_sector_comparison.png` | **Supplementary:** Sector loadings weighted by bloc |

## Key Results

1. **Structural asymmetry:** MERCOSUR-4 extracts approximately 6,100 megatonnes annually, comparable to the EU-27 (5,500 Mt), but concentrated in biomass and metal ores rather than construction minerals.

2. **Six latent components:** NTF identifies agricultural crops, forestry, fishery, non-metallic minerals, livestock and grazing, and metal ores as distinct extraction patterns (K=6, R-squared=0.75).

3. **Agreement impact:** Scenario projections indicate cattle raising (+10.8% above baseline by 2034) and cereal cultivation (+9.7%) as the most affected sectors, with biomass showing the strongest amplification at the MFA category level (+8.5%).

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
Gomez, R. E. (2026). Projecting material extraction under the MERCOSUR-EU
trade agreement.
```

**Zenodo DOI:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18674716.svg)](https://doi.org/10.5281/zenodo.18674716)
