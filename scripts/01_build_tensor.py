"""
01_build_tensor.py — Build 4D tensor from GLORIA MRIO satellite accounts
=========================================================================
Extracts material satellite data for MERCOSUR-4 and EU-27 from TQ parquets.
Produces a 4D tensor: regions(31) × sectors(120) × materials(367) × years(33)

Output:
  data/tensor_materials.nc         (xarray Dataset, NetCDF4)
  data/tensor_summary.parquet      (long-format summary by MFA category)
  data/material_classification.parquet  (367 indicators with MFA labels)
"""

import re
import time

import numpy as np
import pandas as pd
import xarray as xr

from config import (
    META_DIR, TQ_DIR, DATA_DIR,
    ALL_COUNTRIES, MERCOSUR, EU27, BLOC,
    YEAR_RANGE,
)

# ---------------------------------------------------------------------------
# 1. Load metadata
# ---------------------------------------------------------------------------
print("[1/5] Loading metadata...")

regions = pd.read_parquet(META_DIR / "regions.parquet")
sectors = pd.read_parquet(META_DIR / "sectors.parquet")
satellites = pd.read_parquet(META_DIR / "satellites.parquet")
seq_labels = pd.read_parquet(META_DIR / "sequential_labels.parquet")

N_SECTORS = len(sectors)   # 120
N_REGIONS = len(regions)   # 164
BLOCK_SIZE = N_SECTORS * 2  # 240 (industry + product per region)

# ---------------------------------------------------------------------------
# 2. Map region acronyms -> TQ column indices
# ---------------------------------------------------------------------------
print("[2/5] Mapping column indices...")

# Extract region ordering from sequential_labels (every 240 rows = new region)
region_order = []
for i in range(0, len(seq_labels), BLOCK_SIZE):
    label = seq_labels.iloc[i]["sequential_regionsector_labels"]
    match = re.search(r"\(([A-Z]{3})\)", label)
    if match:
        region_order.append(match.group(1))

# Build column index mapping: acronym -> list of industry column names
col_map = {}
for acr in ALL_COUNTRIES:
    if acr not in region_order:
        raise ValueError(f"Country {acr} not found in GLORIA sequential labels")
    idx = region_order.index(acr)
    base = idx * BLOCK_SIZE
    # Industry columns (direct production/extraction)
    industry_cols = [f"c{base + s}" for s in range(N_SECTORS)]
    col_map[acr] = industry_cols

print(f"  Mapped {len(col_map)} countries ({len(MERCOSUR)} MERCOSUR + {len(EU27)} EU-27)")

# ---------------------------------------------------------------------------
# 3. Classify material satellites into MFA categories
# ---------------------------------------------------------------------------
mat_sats = satellites[satellites["sat_head_indicator"] == "Material"].copy()


def classify_material(indicator, lfd_nr):
    """Classify a material indicator into MFA category and subcategory.

    Classification follows EUROSTAT/UNEP-IRP conventions:
      - Biomass (lfd_nr 1-328): crops, crop residues, grazed biomass, wood, fishery
      - Metal ores (lfd_nr 329-343): iron, aluminium, copper, gold, other metals
      - Non-metallic minerals (lfd_nr 344-357): construction, industrial
      - Fossil fuels (lfd_nr 358-367): coal, oil, gas/peat
    """
    ind = indicator.lower()

    # Fossil fuels (lfd_nr 358-367)
    if lfd_nr >= 358:
        if "coal" in ind or "lignite" in ind or "anthracite" in ind or "bituminous" in ind:
            return "Fossil fuels", "Coal"
        elif "oil" in ind or "petroleum" in ind or "tar" in ind:
            return "Fossil fuels", "Oil"
        else:
            # Remaining fossil fuel indicators: natural gas, NGL, peat
            return "Fossil fuels", "Gas and peat"

    # Metal ores (lfd_nr 329-343)
    if 329 <= lfd_nr <= 343:
        if "iron" in ind:
            return "Metal ores", "Iron"
        elif "aluminium" in ind or "bauxite" in ind:
            return "Metal ores", "Aluminium"
        elif "copper" in ind:
            return "Metal ores", "Copper"
        elif "gold" in ind:
            return "Metal ores", "Gold"
        else:
            return "Metal ores", "Other metals"

    # Non-metallic minerals (lfd_nr 344-357)
    if 344 <= lfd_nr <= 357:
        if "sand" in ind or "gravel" in ind or "crushed" in ind or "stone" in ind:
            return "Non-metallic minerals", "Construction minerals"
        else:
            return "Non-metallic minerals", "Industrial minerals"

    # Biomass (lfd_nr 1-327)
    if "residue" in ind:
        return "Biomass", "Crop residues"
    elif "timber" in ind or "wood" in ind:
        return "Biomass", "Wood"
    elif "fish" in ind or "aquatic" in ind:
        return "Biomass", "Fishery and aquatic"
    elif "grazed" in ind:
        return "Biomass", "Grazed biomass"
    else:
        return "Biomass", "Crops"


mat_sats[["mfa_category", "mfa_subcategory"]] = mat_sats.apply(
    lambda r: pd.Series(classify_material(r["sat_indicator"], r["lfd_nr"])), axis=1
)

# Material satellite IDs (1-indexed lfd_nr, matches sat_id in TQ)
mat_sat_ids = mat_sats["lfd_nr"].values
n_materials = len(mat_sat_ids)

print(f"  Material satellites: {n_materials}")
for cat, sub_df in mat_sats.groupby("mfa_category"):
    subs = ", ".join(f"{s}: {n}" for s, n in sub_df.groupby("mfa_subcategory").size().items())
    print(f"    {cat} ({len(sub_df)}): {subs}")

# Save classification for reference
mat_sats.to_parquet(DATA_DIR / "material_classification.parquet", index=False)

# ---------------------------------------------------------------------------
# 4. Extract tensor from TQ parquets
# ---------------------------------------------------------------------------
print("\n[3/5] Extracting tensor from TQ parquets...")

# Determine available years within observed range
available_years = sorted([
    int(p.name.split("=")[1])
    for p in TQ_DIR.iterdir()
    if p.is_dir() and p.name.startswith("year=")
])
years = [y for y in available_years if YEAR_RANGE[0] <= y <= YEAR_RANGE[1]]
print(f"  Years: {min(years)}-{max(years)} ({len(years)} years)")

country_list = list(ALL_COUNTRIES.keys())
sector_names = sectors["sector_names"].values
n_countries = len(country_list)
n_years = len(years)

# Pre-allocate tensor: countries × sectors × materials × years
tensor = np.zeros((n_countries, N_SECTORS, n_materials, n_years), dtype=np.float64)

t0 = time.time()
for yi, year in enumerate(years):
    tq_path = TQ_DIR / f"year={year}" / "data.parquet"

    # Only read the columns we need (sat_id + country industry columns)
    all_cols_needed = ["sat_id"]
    for acr in country_list:
        all_cols_needed.extend(col_map[acr])

    df = pd.read_parquet(tq_path, columns=all_cols_needed)

    # Filter to material satellites only
    df = df[df["sat_id"].isin(mat_sat_ids)].copy()
    df = df.set_index("sat_id").sort_index()

    # Validate row count
    assert len(df) == n_materials, (
        f"Year {year}: expected {n_materials} material rows, got {len(df)}"
    )

    # Extract values for each country
    for ci, acr in enumerate(country_list):
        cols = col_map[acr]
        vals = df[cols].values  # (n_materials, N_SECTORS)
        tensor[ci, :, :, yi] = vals.T  # transpose to (N_SECTORS, n_materials)

    elapsed = time.time() - t0
    eta = (elapsed / (yi + 1)) * (n_years - yi - 1)
    print(f"  [{yi+1:2d}/{n_years}] {year} ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

print(f"  Tensor shape: {tensor.shape}")
print(f"  Total: {tensor.sum():.2e} tonnes")
print(f"  Non-zero: {(tensor > 0).sum():,} / {tensor.size:,} ({(tensor > 0).mean()*100:.1f}%)")

# ---------------------------------------------------------------------------
# 5. Build xarray Dataset
# ---------------------------------------------------------------------------
print("\n[4/5] Building xarray Dataset...")

ds = xr.Dataset(
    {
        "material_flow": xr.DataArray(
            tensor,
            dims=["country", "sector", "material", "year"],
            coords={
                "country": country_list,
                "sector": sector_names,
                "material": mat_sats["sat_indicator"].values,
                "year": years,
            },
            attrs={"units": "tonnes", "description": "Direct material extraction by sector"},
        ),
    },
    attrs={
        "title": "MERCOSUR-EU bilateral material flow tensor",
        "source": "GLORIA MRIO Loop060 satellite accounts (TQ)",
        "countries": f"{len(MERCOSUR)} MERCOSUR + {len(EU27)} EU-27",
        "created": pd.Timestamp.now().isoformat(),
    },
)

# Add coordinate metadata
ds.coords["bloc"] = ("country", [BLOC[c] for c in country_list])
ds.coords["country_name"] = ("country", [ALL_COUNTRIES[c] for c in country_list])
ds.coords["mfa_category"] = ("material", mat_sats["mfa_category"].values)
ds.coords["mfa_subcategory"] = ("material", mat_sats["mfa_subcategory"].values)
ds.coords["sat_id"] = ("material", mat_sat_ids)

nc_path = DATA_DIR / "tensor_materials.nc"
ds.to_netcdf(nc_path)
print(f"  Saved: {nc_path} ({nc_path.stat().st_size / 1e6:.1f} MB)")

# ---------------------------------------------------------------------------
# 6. Summary table (aggregated by MFA category)
# ---------------------------------------------------------------------------
print("\n[5/5] Building summary table...")

records = []
for ci, acr in enumerate(country_list):
    for yi, year in enumerate(years):
        for cat in mat_sats["mfa_category"].unique():
            mask = mat_sats["mfa_category"].values == cat
            total = tensor[ci, :, mask, yi].sum()
            records.append({
                "country": acr,
                "country_name": ALL_COUNTRIES[acr],
                "bloc": BLOC[acr],
                "year": year,
                "mfa_category": cat,
                "tonnes": total,
            })

summary = pd.DataFrame(records)
summary_path = DATA_DIR / "tensor_summary.parquet"
summary.to_parquet(summary_path, index=False)
print(f"  Saved: {summary_path} ({len(summary):,} rows)")

# Verification: print 2022 summary by bloc
check = summary[summary["year"] == max(years)].groupby(["bloc", "mfa_category"])["tonnes"].sum()
check = check.unstack("mfa_category").fillna(0)
check["Total"] = check.sum(axis=1)
print(f"\n  Verification ({max(years)}, megatonnes):")
print((check / 1e6).round(1).to_string())

print("\nDone.")
