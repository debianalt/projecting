"""
04_scenario_projection.py — Trade agreement scenario projection
================================================================
Projects changes in material flows under the MERCOSUR-EU trade agreement
using NTF temporal loadings + bootstrap confidence intervals.

Approach:
  1. Fit trend models to NTF year loadings (1990-2022)
  2. Define tariff shock parameters from the agreement text
  3. Project year loadings under baseline (no agreement) and scenario (agreement)
  4. Bootstrap confidence intervals via resampling residuals
  5. Convert projected loadings back to material tonnes by sector and bloc

Output:
  data/projections.parquet
  data/projections_by_sector.parquet
  figures/fig08_projection_scenarios.png
  figures/fig09_material_impact.png
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from config import (
    DATA_DIR, FIG_DIR, COMP_LABELS, AGREEMENT_SHOCKS, PHASE_IN_YEARS,
    AGREEMENT_YEAR, PROJECTION_END, YEAR_RANGE, N_BOOTSTRAP, RANDOM_SEED,
    MFA_COLORS, MFA_ORDER,
)

np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("[1/7] Loading data...")

loadings = xr.open_dataset(DATA_DIR / "ntf_loadings.nc")
# ds = xr.open_dataset(DATA_DIR / "tensor_materials.nc")  # not used by projection

K = loadings.attrs["optimal_K"]
years = list(loadings.coords["year"].values)
country_codes = list(loadings.coords["country"].values)
blocs = list(loadings.coords["bloc"].values)
sector_names = list(loadings.coords["sector"].values)
mat_subcats = list(loadings.coords["material_subcat"].values)
mfa_cats = list(loadings.coords["mfa_category"].values)

year_loadings = loadings["year_loading"].values     # (n_years, K)
country_loadings = loadings["country_loading"].values  # (n_countries, K)
sector_loadings = loadings["sector_loading"].values    # (n_sectors, K)
material_loadings = loadings["material_loading"].values  # (n_subcats, K)

# Validate that config covers all K components
assert len(COMP_LABELS) >= K, (
    f"COMP_LABELS has {len(COMP_LABELS)} entries but K={K}"
)
assert len(AGREEMENT_SHOCKS) >= K, (
    f"AGREEMENT_SHOCKS has {len(AGREEMENT_SHOCKS)} entries but K={K}"
)

# ---------------------------------------------------------------------------
# 2. Print trade agreement parameters (from config)
# ---------------------------------------------------------------------------
print("[2/7] Trade agreement parameters (from config)...")

for k in range(K):
    shock = AGREEMENT_SHOCKS[k]
    print(f"  C{k+1} ({COMP_LABELS[k]}): MERC +{shock['mercosur_export_boost']*100:.0f}%, "
          f"EU +{shock['eu_export_boost']*100:.0f}%")

# ---------------------------------------------------------------------------
# 3. Fit trend models to year loadings
# ---------------------------------------------------------------------------
print("\n[3/7] Fitting trend models...")

# Projection years start after the last observed year
projection_years = list(range(YEAR_RANGE[1] + 1, PROJECTION_END))
all_years = years + projection_years
n_hist = len(years)
n_proj = len(projection_years)

# Fit quadratic trend to each component's year loading
trend_params = {}
residuals = {}

for k in range(K):
    y = year_loadings[:, k]
    t = np.arange(n_hist)

    # Quadratic fit (captures acceleration/deceleration)
    coeffs = np.polyfit(t, y, 2)
    fitted = np.polyval(coeffs, t)
    resid = y - fitted

    trend_params[k] = coeffs
    residuals[k] = resid

    r2 = 1 - np.sum(resid**2) / np.sum((y - y.mean())**2)
    print(f"  C{k+1}: coeffs={coeffs.round(4)}, trend R²={r2:.3f}")

# ---------------------------------------------------------------------------
# 4. Project with bootstrap
# ---------------------------------------------------------------------------
print(f"\n[4/7] Projecting scenarios with bootstrap (n={N_BOOTSTRAP})...")
# Note: bootstrap residuals are resampled from n_hist observations
# (e.g. 33 years for 1990-2022), which limits the variability captured
# in the confidence intervals to the historical residual distribution.

t_proj = np.arange(n_hist, n_hist + n_proj)
t_all = np.arange(n_hist + n_proj)

# Store projections: baseline and agreement scenario
proj_baseline = np.zeros((N_BOOTSTRAP, n_proj, K))
proj_agreement = np.zeros((N_BOOTSTRAP, n_proj, K))

for b in range(N_BOOTSTRAP):
    for k in range(K):
        # Baseline: trend extrapolation + bootstrapped residuals
        boot_resid = np.random.choice(residuals[k], size=n_proj, replace=True)
        baseline = np.polyval(trend_params[k], t_proj) + boot_resid

        # Agreement scenario: baseline + trade shock
        # The shock amplifies the loading proportionally
        merc_boost = AGREEMENT_SHOCKS[k]["mercosur_export_boost"]
        eu_boost = AGREEMENT_SHOCKS[k]["eu_export_boost"]
        # Weighted average boost (MERCOSUR countries have higher loading in some components)
        merc_mask = np.array([bl == "MERCOSUR" for bl in blocs])
        eu_mask = np.array([bl == "EU27" for bl in blocs])
        merc_weight = country_loadings[merc_mask, k].sum()
        eu_weight = country_loadings[eu_mask, k].sum()
        total_weight = merc_weight + eu_weight

        # Guard against zero total weight (e.g. if a component has no loading)
        if total_weight > 0:
            avg_boost = (merc_boost * merc_weight + eu_boost * eu_weight) / total_weight
        else:
            avg_boost = 0.0

        # Phase-in: agreement takes PHASE_IN_YEARS to fully implement (linear ramp).
        # The ramp starts at AGREEMENT_YEAR; years before that get zero shock.
        years_since_agreement = np.array(projection_years) - AGREEMENT_YEAR
        phase_in = np.clip(years_since_agreement / PHASE_IN_YEARS, 0.0, 1.0)
        agreement = baseline * (1 + avg_boost * phase_in)

        # Ensure non-negativity
        proj_baseline[b, :, k] = np.maximum(baseline, 0)
        proj_agreement[b, :, k] = np.maximum(agreement, 0)

# Compute percentiles
baseline_median = np.median(proj_baseline, axis=0)
baseline_lo = np.percentile(proj_baseline, 5, axis=0)
baseline_hi = np.percentile(proj_baseline, 95, axis=0)

agreement_median = np.median(proj_agreement, axis=0)
agreement_lo = np.percentile(proj_agreement, 5, axis=0)
agreement_hi = np.percentile(proj_agreement, 95, axis=0)

print(f"  Baseline and agreement projections computed ({N_BOOTSTRAP} bootstrap samples)")

# ---------------------------------------------------------------------------
# 5. Fig 08: Projection scenarios
# ---------------------------------------------------------------------------
print("\n[5/7] Plotting projection scenarios (Fig 08)...")

n_rows = (K + 2) // 3
fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4.5 * n_rows), sharex=True,
                         squeeze=False)
axes_flat = axes.flatten()

# Hide surplus axes
for idx in range(K, n_rows * 3):
    axes_flat[idx].set_visible(False)

for k in range(K):
    ax = axes_flat[k]

    # Historical
    ax.plot(years, year_loadings[:, k], "k-", linewidth=2, label="Observed")

    # Baseline projection
    ax.plot(projection_years, baseline_median[:, k], "--", color="#7f8c8d", linewidth=1.5,
            label="Baseline (no agreement)")
    ax.fill_between(projection_years, baseline_lo[:, k], baseline_hi[:, k],
                    color="#7f8c8d", alpha=0.15)

    # Agreement scenario
    ax.plot(projection_years, agreement_median[:, k], "-", color="#e74c3c", linewidth=2,
            label="Agreement scenario")
    ax.fill_between(projection_years, agreement_lo[:, k], agreement_hi[:, k],
                    color="#e74c3c", alpha=0.15)

    # Vertical line at agreement start
    ax.axvline(AGREEMENT_YEAR, color="#2c3e50", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.annotate("Agreement\nenters force", xy=(AGREEMENT_YEAR, ax.get_ylim()[1] * 0.95),
                fontsize=7, ha="center", color="#2c3e50")

    ax.set_title(f"C{k+1}: {COMP_LABELS[k]}", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.2)
    if k >= K - 3:
        ax.set_xlabel("Year")
    if k == 0:
        ax.legend(fontsize=7, frameon=False, loc="upper left")

plt.suptitle(f"Projected NTF temporal loadings: baseline vs. trade agreement scenario\n"
             f"(shaded: 90% bootstrap CI, n={N_BOOTSTRAP:,})",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig08_projection_scenarios.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: fig08_projection_scenarios.png")

# ---------------------------------------------------------------------------
# 6. Material impact in TONNES (round-2 revision)
# ---------------------------------------------------------------------------
# The agreement effect is a proportional change in *tonnes*, not in the
# log-scale reconstruction index. Applying the shock to the log-scale loading
# and reading the resulting index change as a tonnage change misstates the
# effect by roughly an order of magnitude (reviewer comment 1, round 2). The
# corrected computation lives in scenario.py: the shock scales each component's
# contribution to extraction in tonnes, the perturbed tensor is back-transformed
# before aggregation, and the reported effect is a genuine tonnage percentage.
print("\n[6/7] Computing material impact in tonnes (scenario.py)...")

import scenario as S

ntf = S.load_ntf()
_, eff_load, _, _ = S.baseline_loadings(ntf)
central = S.central_delta(ntf)

cat_effect = S.tonnes_effect(ntf, eff_load, central)   # symmetric across blocs
cat_df = pd.DataFrame(
    [{"mfa_category": c, "agreement_effect_pct": round(cat_effect[c], 3)}
     for c in S.MFA3]
)
cat_df.to_parquet(DATA_DIR / "projections.parquet", index=False)
print("  Central-scenario MFA-category effect (tonnes, % vs baseline):")
print(cat_df.to_string(index=False))
print("  Saved: projections.parquet")

# ---------------------------------------------------------------------------
# 7. Sector-level impact in TONNES
# ---------------------------------------------------------------------------
print("\n[7/7] Computing sector-level impact in tonnes...")

sector_effect = S.sector_effects(ntf, central)         # loading-weighted, tonnes
sector_df = pd.DataFrame({
    "sector_idx": range(len(ntf.sectors)),
    "sector_name": ntf.sectors,
    "agreement_effect_pct": np.round(sector_effect, 3),
})
sector_df.to_parquet(DATA_DIR / "projections_by_sector.parquet", index=False)
print("  Saved: projections_by_sector.parquet")

top = sector_df.nlargest(5, "agreement_effect_pct")
print("\n  Top 5 sectors by agreement effect (tonnes %):")
for _, r in top.iterrows():
    print(f"    +{r['agreement_effect_pct']:.1f}%  {r['sector_name'][:50]}")

print("\nNote: the publication scenario figures (Fig. 5-7) are produced by")
print("07_regenerate_all_figures.py; fig08 here is an intermediate draft.")
print("\nDone.")
