# Codebook: Variable and File Definitions

## tensor_summary.parquet

Long-format summary of material extraction aggregated by MFA category, country, and year.

| Variable | Description | Unit | Source |
|----------|-------------|------|--------|
| iso3 | ISO 3166-1 alpha-3 country code | - | GLORIA |
| country | Country name | - | GLORIA |
| bloc | Economic bloc (MERCOSUR, EU27) | - | Assigned |
| year | Year | YYYY | GLORIA |
| mfa_category | MFA broad category | - | EUROSTAT/UNEP-IRP |
| value | Material extraction | kilotonnes | GLORIA satellite accounts |

MFA categories follow EUROSTAT/UNEP-IRP classification:
- **Biomass**: agricultural crops, livestock, forestry, fishery
- **Metal ores**: ferrous, non-ferrous, precious metals
- **Non-metallic minerals**: construction minerals, industrial minerals
- **Fossil fuels**: coal, oil, natural gas

## material_classification.parquet

Classification of the 367 material satellite indicators from GLORIA into MFA categories and subcategories.

| Variable | Description |
|----------|-------------|
| satellite_idx | Index in GLORIA satellite matrix |
| satellite_name | Original GLORIA satellite label |
| mfa_category | Broad MFA category (Biomass, Metal ores, Non-metallic minerals, Fossil fuels) |
| mfa_subcategory | Detailed subcategory (15 groups) |

## ntf_loadings.nc (NetCDF4)

NTF component loadings from the decomposition with K=6 components.

| Dimension | Description | Size |
|-----------|-------------|------|
| country | ISO3 country codes | 31 |
| sector | GLORIA sector indices | 120 |
| material_subcat | Material subcategories | 15 |
| year | Years | 33 (1990-2022) |
| component | NTF components (0-5) | 6 |

Variables:
- `country_loadings`: Country factor matrix (31 x 6)
- `sector_loadings`: Sector factor matrix (120 x 6)
- `material_loadings`: Material subcategory factor matrix (15 x 6)
- `year_loadings`: Temporal factor matrix (33 x 6)

## ntf_diagnostics.parquet

Rank selection diagnostics for K = 2 to 10.

| Variable | Description |
|----------|-------------|
| K | Number of components |
| R2 | Variance explained (R-squared) |
| relative_error | Frobenius norm relative error |
| sparsity | Average sparsity of factor matrices |
| marginal_gain | Incremental R-squared from adding one component |

## projections.parquet

Central-scenario agreement effect by MFA category, at 2034, in tonnage terms
(percentage difference between the agreement and baseline paths). Computed by
`04_scenario_projection.py` via `scenario.py`: the shock is applied to
extraction in tonnes and the perturbed tensor is back-transformed from the
log(1 + x) space before aggregation, so the effect is a genuine tonnage change
(not a change in a log-scale index). Effects are symmetric across blocs under
the central calibration.

| Variable | Description | Unit |
|----------|-------------|------|
| mfa_category | MFA category (Biomass, Metal ores, Non-metallic minerals) | - |
| agreement_effect_pct | Agreement vs baseline effect at 2034 | percent |

## projections_by_sector.parquet

Sector-level agreement effect on material extraction, in tonnage terms, computed
as the component-loading-weighted mean of the per-component shocks (`scenario.py`).

| Variable | Description | Unit |
|----------|-------------|------|
| sector_idx | GLORIA sector index (0-based) | - |
| sector_name | Sector label | - |
| agreement_effect_pct | Agreement vs baseline effect at 2034 | percent |

## ne_110m_admin_0_countries.zip

Natural Earth 1:110m country boundaries shapefile used for choropleth maps. Source: https://www.naturalearthdata.com/

## NTF Component Definitions

Components identified by the Non-negative Tensor Factorisation (K=6, R-squared=0.75):

| Component | Label | Dominant materials | Key countries |
|-----------|-------|-------------------|---------------|
| C1 | Agricultural crops | Cereals, sugar crops, vegetables | BRA, FRA, ARG |
| C2 | Forestry & wood | Roundwood, wood fuel | BRA, SWE, DEU |
| C3 | Fishery & aquatic | Marine fish, crustaceans | ARG, DNK, BRA |
| C4 | Non-metallic minerals | Sand, gravel, limestone | BRA, ESP, DEU |
| C5 | Livestock & grazing | Cattle, fodder crops, grazing | BRA, ARG, FRA |
| C6 | Metal ores | Iron ore, copper, bauxite | BRA, FIN, ESP |

## Analysis Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| YEAR_RANGE | 1990-2022 | Observed data window |
| OPTIMAL_K | 6 | NTF rank (components) |
| N_BOOTSTRAP | 1000 | Bootstrap replications for projection CI |
| RANDOM_SEED | 42 | Reproducibility seed |
| AGREEMENT_YEAR | 2027 | Assumed year agreement enters into force |
| PHASE_IN_YEARS | 5 | Linear ramp for tariff concessions |
| PROJECTION_END | 2035 | End of projection horizon (exclusive) |
