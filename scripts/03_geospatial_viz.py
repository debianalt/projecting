"""
03_geospatial_viz.py — Geospatial visualisations of NTF components
===================================================================
Creates:
  fig03: Choropleth maps of NTF country loadings (one map per component)
  fig04: Temporal evolution by bloc (MERCOSUR vs EU-27 weighted time series)
  fig05: Material composition heatmap (subcategory × component)
  fig06: Top bilateral sector loadings weighted by bloc (grouped bar chart)
  fig07: Historical material extraction by MFA category and bloc (stacked area)

Requires: geopandas, matplotlib, seaborn
"""

import urllib.request

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

from config import (
    DATA_DIR, FIG_DIR, COMP_LABELS, BLOC_COLORS, MFA_COLORS, MFA_ORDER,
)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("[1/6] Loading data...")

loadings = xr.open_dataset(DATA_DIR / "ntf_loadings.nc")
tensor_summary = pd.read_parquet(DATA_DIR / "tensor_summary.parquet")
ds = xr.open_dataset(DATA_DIR / "tensor_materials.nc")

K = loadings.attrs["optimal_K"]
R2 = loadings.attrs["final_R2"]

# Validate that COMP_LABELS covers all K components
assert len(COMP_LABELS) >= K, (
    f"COMP_LABELS has {len(COMP_LABELS)} entries but K={K}"
)

country_codes = list(loadings.coords["country"].values)
blocs = list(loadings.coords["bloc"].values)
years = list(loadings.coords["year"].values)
components = list(loadings.coords["component"].values)

# Build string-keyed comp_labels for use with "C1", "C2", ... keys
comp_labels = {f"C{k+1}": COMP_LABELS[k] for k in range(K)}

# Load world geometries (Natural Earth 110m countries)
NE_URL = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
ne_cache = DATA_DIR / "ne_110m_admin_0_countries.zip"
if not ne_cache.exists():
    try:
        print("  Downloading Natural Earth countries...")
        urllib.request.urlretrieve(NE_URL, ne_cache)
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(
            f"Failed to download Natural Earth shapefile from {NE_URL}: {exc}"
        ) from exc
world = gpd.read_file(ne_cache)

# Normalise ISO codes (column is ISO_A3 in NE 110m)
world["ISO_A3"] = world["ISO_A3"].replace({"-99": None, "-099": None})
# Fix countries with missing ISO codes using NAME_EN
fixes = {"France": "FRA", "Norway": "NOR", "Kosovo": "XKX"}
for name, code in fixes.items():
    world.loc[world["NAME_EN"] == name, "ISO_A3"] = code

print(f"  Loaded {len(world)} world polygons")
print(f"  K={K}, R²={R2:.4f}")

# ---------------------------------------------------------------------------
# 2. Fig 03: Choropleth maps of country loadings
# ---------------------------------------------------------------------------
print("\n[2/6] Creating choropleth maps (Fig 03)...")

# Regions to zoom: South America + Europe
sa_bounds = [-82, -56, -20, 15]   # [minx, miny, maxx, maxy]
eu_bounds = [-12, 34, 35, 72]

fig = plt.figure(figsize=(18, 4 * ((K + 1) // 2)))
gs = gridspec.GridSpec((K + 1) // 2, 4, width_ratios=[1, 1, 1, 1],
                       wspace=0.05, hspace=0.25)

for k in range(K):
    row = k // 2
    col_offset = (k % 2) * 2

    # Merge loadings with world geometries
    loading_vals = loadings["country_loading"][:, k].values
    df_load = pd.DataFrame({
        "ISO_A3": country_codes,
        "loading": loading_vals,
        "bloc": blocs,
    })
    merged = world.merge(df_load, on="ISO_A3", how="left")

    vmax = loading_vals.max() * 1.05
    cmap = plt.cm.YlOrRd

    # South America panel
    ax_sa = fig.add_subplot(gs[row, col_offset])
    world.plot(ax=ax_sa, color="#f0f0f0", edgecolor="#cccccc", linewidth=0.3)
    merged[merged["loading"].notna()].plot(
        ax=ax_sa, column="loading", cmap=cmap, vmin=0, vmax=vmax,
        edgecolor="#333333", linewidth=0.5, legend=False,
    )
    ax_sa.set_xlim(sa_bounds[0], sa_bounds[2])
    ax_sa.set_ylim(sa_bounds[1], sa_bounds[3])
    ax_sa.set_title(f"C{k+1}: {comp_labels.get(f'C{k+1}', '')}\nMERCOSUR", fontsize=9, fontweight="bold")
    ax_sa.axis("off")

    # Europe panel
    ax_eu = fig.add_subplot(gs[row, col_offset + 1])
    world.plot(ax=ax_eu, color="#f0f0f0", edgecolor="#cccccc", linewidth=0.3)
    merged[merged["loading"].notna()].plot(
        ax=ax_eu, column="loading", cmap=cmap, vmin=0, vmax=vmax,
        edgecolor="#333333", linewidth=0.5, legend=False,
    )
    ax_eu.set_xlim(eu_bounds[0], eu_bounds[2])
    ax_eu.set_ylim(eu_bounds[1], eu_bounds[3])
    ax_eu.set_title(f"C{k+1}: {comp_labels.get(f'C{k+1}', '')}\nEU-27", fontsize=9, fontweight="bold")
    ax_eu.axis("off")

    # Add colorbar to Europe panel
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_eu, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=7)

plt.suptitle(
    f"NTF country loadings (K={K}, R²={R2:.2f})",
    fontsize=13, fontweight="bold", y=1.01,
)
plt.savefig(FIG_DIR / "fig03_choropleth_loadings.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: fig03_choropleth_loadings.png")

# ---------------------------------------------------------------------------
# 3. Fig 04: Temporal evolution by bloc
# ---------------------------------------------------------------------------
print("\n[3/6] Creating temporal evolution plot (Fig 04)...")

year_loadings = loadings["year_loading"].values    # (n_years, K)
country_loadings = loadings["country_loading"].values  # (n_countries, K)
# Derive shape comment from actual data
n_years_actual, K_actual = year_loadings.shape
n_countries_actual, _ = country_loadings.shape

# Weighted temporal trends by bloc — dynamic subplot grid for arbitrary K
n_rows = (K + 2) // 3
fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows), sharex=True,
                         squeeze=False)
axes_flat = axes.flatten()

# Hide surplus axes if K is not a multiple of 3
for idx in range(K, n_rows * 3):
    axes_flat[idx].set_visible(False)

for k in range(K):
    ax = axes_flat[k]
    for bloc_name, color in BLOC_COLORS.items():
        # Countries in this bloc
        mask = np.array([b == bloc_name for b in blocs])
        # Bloc weight is the mean country loading: mean() is used rather
        # than sum() so that the weighted trend is comparable across blocs
        # of different sizes (4 MERCOSUR countries vs 27 EU countries).
        bloc_weight = country_loadings[mask, k].mean()
        weighted_trend = year_loadings[:, k] * bloc_weight

        ax.plot(years, weighted_trend, "-", color=color, linewidth=2, label=bloc_name)
        ax.fill_between(years, 0, weighted_trend, color=color, alpha=0.1)

    ax.set_title(f"C{k+1}: {comp_labels.get(f'C{k+1}', '')}", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.2)
    if k >= K - 3:
        ax.set_xlabel("Year")
    if k == 0:
        ax.legend(frameon=False, fontsize=9)

plt.suptitle("Temporal evolution of NTF components by bloc", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig04_temporal_blocs.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: fig04_temporal_blocs.png")

# ---------------------------------------------------------------------------
# 4. Fig 05: Material composition heatmap
# ---------------------------------------------------------------------------
print("\n[4/6] Creating material composition heatmap (Fig 05)...")

mat_loadings = loadings["material_loading"].values  # (n_subcats, K)
mat_subcats = list(loadings.coords["material_subcat"].values)
mfa_cats = list(loadings.coords["mfa_category"].values)

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    mat_loadings,
    xticklabels=[f"C{k+1}\n{comp_labels.get(f'C{k+1}', '')}" for k in range(K)],
    yticklabels=mat_subcats,
    cmap="YlOrRd",
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    ax=ax,
)
ax.set_title(f"Material subcategory loadings on NTF components (K={K})", fontsize=12)
ax.set_ylabel("Material subcategory")
ax.set_xlabel("Component")

# Color-code y-axis labels by MFA category
for i, (label, cat) in enumerate(zip(ax.get_yticklabels(), mfa_cats)):
    label.set_color(MFA_COLORS.get(cat, "black"))
    label.set_fontweight("bold")

plt.tight_layout()
plt.savefig(FIG_DIR / "fig05_material_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: fig05_material_heatmap.png")

# ---------------------------------------------------------------------------
# 5. Fig 06: Sector loading comparison MERCOSUR vs EU
# ---------------------------------------------------------------------------
print("\n[5/6] Creating sector comparison plot (Fig 06)...")

sector_loadings = loadings["sector_loading"].values  # (n_sectors, K)
sector_names = list(loadings.coords["sector"].values)

# For each component, show top 10 sectors with MERCOSUR vs EU weighted values
# Dynamic subplot grid for arbitrary K
n_rows = (K + 2) // 3
fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6 * n_rows), squeeze=False)
axes_flat = axes.flatten()

# Hide surplus axes if K is not a multiple of 3
for idx in range(K, n_rows * 3):
    axes_flat[idx].set_visible(False)

for k in range(K):
    ax = axes_flat[k]

    # Sector loading
    s_load = sector_loadings[:, k]

    # Top 10 sectors
    top_idx = np.argsort(s_load)[::-1][:10]

    # Weighted by average country loading per bloc
    merc_weight = country_loadings[np.array([b == "MERCOSUR" for b in blocs]), k].mean()
    eu_weight = country_loadings[np.array([b == "EU27" for b in blocs]), k].mean()

    y_pos = np.arange(len(top_idx))
    bar_h = 0.35

    ax.barh(y_pos - bar_h/2, s_load[top_idx] * merc_weight, bar_h,
            color=BLOC_COLORS["MERCOSUR"], label="MERCOSUR" if k == 0 else None, alpha=0.8)
    ax.barh(y_pos + bar_h/2, s_load[top_idx] * eu_weight, bar_h,
            color=BLOC_COLORS["EU27"], label="EU-27" if k == 0 else None, alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([sector_names[i][:40] for i in top_idx], fontsize=7)
    ax.invert_yaxis()
    ax.set_title(f"C{k+1}: {comp_labels.get(f'C{k+1}', '')}", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="x")

    if k == 0:
        ax.legend(frameon=False, fontsize=9)

plt.suptitle("Sector loadings weighted by bloc (top 10 per component)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig06_sector_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: fig06_sector_comparison.png")

# ---------------------------------------------------------------------------
# 6. Fig 07: Historical material flows by MFA category and bloc
# ---------------------------------------------------------------------------
print("\n[6/6] Creating historical material flow trends (Fig 07)...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

for bi, (bloc_name, ax) in enumerate(zip(["MERCOSUR", "EU27"], axes)):
    bloc_data = tensor_summary[tensor_summary["bloc"] == bloc_name]
    pivot = bloc_data.groupby(["year", "mfa_category"])["tonnes"].sum().unstack("mfa_category")
    pivot = pivot[MFA_ORDER] / 1e6  # megatonnes

    pivot.plot.area(ax=ax, color=[MFA_COLORS[c] for c in MFA_ORDER], alpha=0.7, linewidth=0)
    ax.set_title(bloc_name, fontsize=12, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Megatonnes" if bi == 0 else "")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.2)

plt.suptitle("Material extraction by MFA category (1990-2022)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig07_historical_trends.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: fig07_historical_trends.png")

print("\nAll figures generated.")
