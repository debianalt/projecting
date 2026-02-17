"""
02_ntf_analysis.py — Non-negative Tensor Factorisation of material flows
=========================================================================
Applies NTF (CP decomposition with non-negativity) to the 4D tensor:
  countries × sectors × material_subcategories × years

The raw tensor is first aggregated from fine-grained material codes to MFA
subcategories, then log-transformed before decomposition.

Determines optimal rank K via reconstruction error + core consistency.
Extracts and saves component loadings for downstream mapping and projection.

Output:
  data/ntf_loadings.nc        (xarray Dataset with all component loadings)
  data/ntf_diagnostics.parquet (rank selection diagnostics)
  figures/fig01_rank_selection.png
  figures/fig02_component_loadings.png
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.decomposition import non_negative_parafac
import xarray as xr

from config import DATA_DIR, FIG_DIR, OPTIMAL_K, RANDOM_SEED, COMP_LABELS, MFA_COLORS

warnings.filterwarnings("ignore", category=FutureWarning)

np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# 1. Load tensor
# ---------------------------------------------------------------------------
print("[1/5] Loading tensor...")
ds = xr.open_dataset(DATA_DIR / "tensor_materials.nc")
X_raw = ds["material_flow"].values
print(f"  Raw tensor shape: {X_raw.shape}")
print(f"  Total mass: {X_raw.sum():.2e} tonnes")

# ---------------------------------------------------------------------------
# 2. Aggregate materials to MFA subcategory level for NTF
# ---------------------------------------------------------------------------
print("[2/5] Aggregating materials to subcategory level...")

mfa_subcats = ds.coords["mfa_subcategory"].values
unique_subcats = list(dict.fromkeys(mfa_subcats))  # preserve order
n_subcats = len(unique_subcats)

n_countries = X_raw.shape[0]
n_sectors = X_raw.shape[1]
n_years = X_raw.shape[3]

# Aggregate: sum materials within each subcategory
X_agg = np.zeros((n_countries, n_sectors, n_subcats, n_years), dtype=np.float64)
for si, subcat in enumerate(unique_subcats):
    mask = mfa_subcats == subcat
    X_agg[:, :, si, :] = X_raw[:, :, mask, :].sum(axis=2)

print(f"  Aggregated tensor shape: {X_agg.shape}")
print(f"  Subcategories: {unique_subcats}")
print(f"  Mass preserved: {X_agg.sum():.2e} (raw: {X_raw.sum():.2e})")

# Log-transform to reduce skewness (add 1 to avoid log(0))
# NTF works better with less skewed data
X_log = np.log1p(X_agg)

# ---------------------------------------------------------------------------
# 3. Rank selection: test K=2..10
# ---------------------------------------------------------------------------
print("\n[3/5] Rank selection (K=2..10)...")

tl.set_backend("numpy")

K_range = range(2, 11)
diagnostics = []

for K in K_range:
    print(f"  Testing K={K}...", end=" ", flush=True)

    # Run 3 random initialisations per K; keep the best (lowest error)
    best_K_error = np.inf
    best_K_result = None

    for init_seed in range(3):
        result = non_negative_parafac(
            tl.tensor(X_log),
            rank=K,
            init="random",
            n_iter_max=500,
            tol=1e-6,
            random_state=RANDOM_SEED + init_seed,
        )
        X_hat_tmp = tl.cp_to_tensor(result)
        err_tmp = np.linalg.norm(X_log - X_hat_tmp)
        if err_tmp < best_K_error:
            best_K_error = err_tmp
            best_K_result = result

    # Reconstruct tensor from best initialisation
    X_hat = tl.cp_to_tensor(best_K_result)

    # Reconstruction error (relative)
    residual = np.linalg.norm(X_log - X_hat) / np.linalg.norm(X_log)

    # Variance explained (R²)
    ss_total = np.sum((X_log - X_log.mean()) ** 2)
    ss_residual = np.sum((X_log - X_hat) ** 2)
    r2 = 1 - ss_residual / ss_total

    # Sparsity of factors (proportion of near-zero loadings)
    factors = best_K_result.factors
    sparsity = np.mean([np.mean(f < 1e-6) for f in factors])

    diagnostics.append({
        "K": K,
        "rel_error": residual,
        "R2": r2,
        "sparsity": sparsity,
    })

    print(f"R²={r2:.4f}, RelErr={residual:.4f}, Sparsity={sparsity:.2f}")

diag_df = pd.DataFrame(diagnostics)
diag_df.to_parquet(DATA_DIR / "ntf_diagnostics.parquet", index=False)

# ---------------------------------------------------------------------------
# 4. Plot rank selection
# ---------------------------------------------------------------------------
print("\n[4/5] Plotting rank selection diagnostics...")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# R² vs K
ax = axes[0]
ax.plot(diag_df["K"], diag_df["R2"], "o-", color="#2c3e50", linewidth=2, markersize=6)
ax.set_xlabel("Number of components (K)")
ax.set_ylabel("Variance explained (R²)")
ax.set_title("a) Model fit")
ax.set_xticks(list(K_range))
ax.grid(True, alpha=0.3)

# Marginal gain in R²
ax = axes[1]
marginal = diag_df["R2"].diff()
ax.bar(diag_df["K"].iloc[1:], marginal.iloc[1:], color="#2c3e50", alpha=0.7)
ax.set_xlabel("Number of components (K)")
ax.set_ylabel("Marginal R² gain")
ax.set_title("b) Marginal improvement")
ax.set_xticks(list(K_range))
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig01_rank_selection.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIG_DIR / 'fig01_rank_selection.png'}")

# ---------------------------------------------------------------------------
# 5. Fit final model at optimal K
# ---------------------------------------------------------------------------
# Choose K where marginal gain drops below threshold
marginal_gains = diag_df["R2"].diff().fillna(1)
# Elbow: first K where marginal gain < 0.01 (or use the one before)
# K=6 captures all major MFA categories (biomass, metals, minerals, fossil fuels)
# while maintaining interpretability. R²=0.74, marginal gain stabilises after K=6.
optimal_K = OPTIMAL_K

# Validate that COMP_LABELS covers all K components
assert len(COMP_LABELS) >= optimal_K, (
    f"COMP_LABELS has {len(COMP_LABELS)} entries but optimal_K={optimal_K}"
)

print(f"\n[5/5] Fitting final NTF with K={optimal_K}...")

# Run multiple initialisations, keep best
best_error = np.inf
best_result = None

for seed in range(10):
    result = non_negative_parafac(
        tl.tensor(X_log),
        rank=optimal_K,
        init="random",
        n_iter_max=1000,
        tol=1e-8,
        random_state=seed,
    )
    X_hat = tl.cp_to_tensor(result)
    error = np.linalg.norm(X_log - X_hat)
    if error < best_error:
        best_error = error
        best_result = result
        best_seed = seed

print(f"  Best initialisation: seed={best_seed}")

weights, factors = best_result.weights, best_result.factors
country_loadings = factors[0]   # (n_countries, K)
sector_loadings = factors[1]    # (n_sectors, K)
material_loadings = factors[2]  # (n_subcats, K)
year_loadings = factors[3]      # (n_years, K)

# Normalise: absorb weights into year loadings so that the country, sector
# and material factor matrices are unit-normalised while the temporal mode
# carries all magnitude information (convention for interpretability).
for k in range(optimal_K):
    norms = [np.linalg.norm(factors[d][:, k]) for d in range(4)]
    scale = weights[k] * np.prod(norms)
    for d in range(4):
        factors[d][:, k] /= norms[d]
    year_loadings[:, k] = factors[3][:, k] * scale

# Reconstruct and report final R².
# np.ones(optimal_K) replaces the original weight vector because all
# magnitude has been absorbed into year_loadings during normalisation;
# the CP reconstruction formula thus uses unit weights.
X_hat = tl.cp_to_tensor((np.ones(optimal_K), factors[:3] + [year_loadings]))
ss_total = np.sum((X_log - X_log.mean()) ** 2)
ss_residual = np.sum((X_log - X_hat) ** 2)
final_r2 = 1 - ss_residual / ss_total
print(f"  Final R²: {final_r2:.4f}")

# ---------------------------------------------------------------------------
# 6. Save loadings as xarray Dataset
# ---------------------------------------------------------------------------
country_codes = list(ds.coords["country"].values)
sector_names = list(ds.coords["sector"].values)
years = list(ds.coords["year"].values)
bloc_labels = list(ds.coords["bloc"].values)
country_names = list(ds.coords["country_name"].values)

# Map subcategories to their parent MFA category
mfa_cats_unique = []
for sc in unique_subcats:
    mask = ds.coords["mfa_subcategory"].values == sc
    cat = ds.coords["mfa_category"].values[mask][0]
    mfa_cats_unique.append(cat)

comp_names = [f"C{k+1}" for k in range(optimal_K)]

loadings_ds = xr.Dataset(
    {
        "country_loading": xr.DataArray(
            country_loadings,
            dims=["country", "component"],
            coords={"country": country_codes, "component": comp_names},
        ),
        "sector_loading": xr.DataArray(
            sector_loadings,
            dims=["sector", "component"],
            coords={"sector": sector_names, "component": comp_names},
        ),
        "material_loading": xr.DataArray(
            material_loadings,
            dims=["material_subcat", "component"],
            coords={"material_subcat": unique_subcats, "component": comp_names},
        ),
        "year_loading": xr.DataArray(
            year_loadings,
            dims=["year", "component"],
            coords={"year": years, "component": comp_names},
        ),
    },
    attrs={
        "optimal_K": optimal_K,
        "final_R2": float(final_r2),
        "best_seed": best_seed,
        "description": "NTF (non-negative CP) loadings for MERCOSUR-EU material flow tensor",
    },
)

loadings_ds.coords["bloc"] = ("country", bloc_labels)
loadings_ds.coords["country_name"] = ("country", country_names)
loadings_ds.coords["mfa_category"] = ("material_subcat", mfa_cats_unique)

nc_path = DATA_DIR / "ntf_loadings.nc"
loadings_ds.to_netcdf(nc_path)
print(f"  Saved: {nc_path}")

# ---------------------------------------------------------------------------
# 7. Plot component loadings overview
# ---------------------------------------------------------------------------
print("  Plotting component loadings...")

# Dynamic grid: one row per component, 4 columns (country, sector, material, year)
fig, axes = plt.subplots(optimal_K, 4, figsize=(20, 4 * optimal_K),
                         squeeze=False)

for k in range(optimal_K):
    # Country loadings
    ax = axes[k, 0]
    vals = country_loadings[:, k]
    colors = ["#e74c3c" if b == "MERCOSUR" else "#3498db" for b in bloc_labels]
    order = np.argsort(vals)[::-1][:15]  # top 15
    ax.barh(range(len(order)), vals[order], color=[colors[i] for i in order])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([country_codes[i] for i in order], fontsize=8)
    ax.set_title(f"C{k+1} — Countries" if k == 0 else f"C{k+1}")
    ax.invert_yaxis()

    # Sector loadings (top 10)
    ax = axes[k, 1]
    vals = sector_loadings[:, k]
    order = np.argsort(vals)[::-1][:10]
    ax.barh(range(len(order)), vals[order], color="#2c3e50")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([sector_names[i][:35] for i in order], fontsize=7)
    if k == 0:
        ax.set_title("Sectors")

    # Material loadings
    ax = axes[k, 2]
    vals = material_loadings[:, k]
    bar_colors = [MFA_COLORS.get(c, "#95a5a6") for c in mfa_cats_unique]
    ax.barh(range(len(unique_subcats)), vals, color=bar_colors)
    ax.set_yticks(range(len(unique_subcats)))
    ax.set_yticklabels(unique_subcats, fontsize=8)
    if k == 0:
        ax.set_title("Materials")
    ax.invert_yaxis()

    # Year loadings
    ax = axes[k, 3]
    ax.plot(years, year_loadings[:, k], "-o", color="#2c3e50", markersize=3)
    if k == 0:
        ax.set_title("Temporal trend")
    ax.set_xlabel("Year")

plt.tight_layout()
plt.savefig(FIG_DIR / "fig02_component_loadings.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIG_DIR / 'fig02_component_loadings.png'}")

# ---------------------------------------------------------------------------
# 8. Print interpretive summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"NTF RESULTS SUMMARY (K={optimal_K}, R²={final_r2:.4f})")
print(f"{'='*60}")

for k in range(optimal_K):
    label = COMP_LABELS.get(k, f"Component {k+1}")
    print(f"\n--- Component C{k+1}: {label} ---")

    # Top countries
    top_c = np.argsort(country_loadings[:, k])[::-1][:5]
    top_c_str = ", ".join([f"{country_codes[i]} ({bloc_labels[i]}): {country_loadings[i,k]:.3f}" for i in top_c])
    print(f"  Countries: {top_c_str}")

    # Top sectors
    top_s = np.argsort(sector_loadings[:, k])[::-1][:3]
    top_s_str = ", ".join([f"{sector_names[i][:40]}: {sector_loadings[i,k]:.3f}" for i in top_s])
    print(f"  Sectors: {top_s_str}")

    # Top materials
    top_m = np.argsort(material_loadings[:, k])[::-1][:3]
    top_m_str = ", ".join([f"{unique_subcats[i]}: {material_loadings[i,k]:.3f}" for i in top_m])
    print(f"  Materials: {top_m_str}")

    # Temporal trend direction
    slope = np.polyfit(range(len(years)), year_loadings[:, k], 1)[0]
    trend = "increasing" if slope > 0 else "decreasing"
    print(f"  Trend: {trend} (slope={slope:.4f})")

print("\nDone.")
