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
# 6. Fig 09: Material impact by bloc and MFA category
# ---------------------------------------------------------------------------
print("\n[6/7] Computing material impact and plotting (Fig 09)...")

# Convert projected loadings back to material tonnes
# Tensor reconstruction: X_ijk_t = sum_k (country_i * sector_j * material_m * year_t)
# Total change in material tonnes for a bloc = sum over countries, sectors, materials

last_obs_year = YEAR_RANGE[1]
last_proj_year = projection_years[-1]


def compute_material_tonnes(year_load_vector):
    """Reconstruct total material tonnes by bloc from NTF loadings."""
    results = {}
    for bloc_name in ["MERCOSUR", "EU27"]:
        mask = np.array([b == bloc_name for b in blocs])
        total = 0
        for k in range(K):
            c_sum = country_loadings[mask, k].sum()
            s_sum = sector_loadings[:, k].sum()
            m_sum = material_loadings[:, k].sum()
            y_val = year_load_vector[k]
            total += c_sum * s_sum * m_sum * y_val
        results[bloc_name] = total
    return results

# 2022 values (last observed year)
hist_2022 = compute_material_tonnes(year_loadings[-1, :])

# 2034 baseline and agreement (median)
base_2034 = compute_material_tonnes(baseline_median[-1, :])
agree_2034 = compute_material_tonnes(agreement_median[-1, :])

# Decompose by MFA category and bloc at 2034
impact_records = []
for bloc_name in ["MERCOSUR", "EU27"]:
    mask_c = np.array([b == bloc_name for b in blocs])
    for mi, (subcat, mfa_cat) in enumerate(zip(mat_subcats, mfa_cats)):
        for scenario_name, year_vec in [
            (f"Baseline {last_proj_year}", baseline_median[-1, :]),
            (f"Agreement {last_proj_year}", agreement_median[-1, :]),
            (f"Observed {last_obs_year}", year_loadings[-1, :]),
        ]:
            total = 0
            for k in range(K):
                c_sum = country_loadings[mask_c, k].sum()
                s_sum = sector_loadings[:, k].sum()
                m_val = material_loadings[mi, k]
                y_val = year_vec[k]
                total += c_sum * s_sum * m_val * y_val
            impact_records.append({
                "bloc": bloc_name,
                "mfa_category": mfa_cat,
                "material_subcat": subcat,
                "scenario": scenario_name,
                "log_tonnes_index": total,
            })

impact_df = pd.DataFrame(impact_records)

# Aggregate by bloc and MFA category
agg = impact_df.groupby(["bloc", "mfa_category", "scenario"])["log_tonnes_index"].sum().reset_index()
pivot = agg.pivot_table(index=["bloc", "mfa_category"], columns="scenario",
                        values="log_tonnes_index").reset_index()

# Compute percentage change
pivot["change_baseline_%"] = ((pivot[f"Baseline {last_proj_year}"] - pivot[f"Observed {last_obs_year}"]) /
                               pivot[f"Observed {last_obs_year}"] * 100)
pivot["change_agreement_%"] = ((pivot[f"Agreement {last_proj_year}"] - pivot[f"Observed {last_obs_year}"]) /
                                pivot[f"Observed {last_obs_year}"] * 100)
pivot["agreement_effect_%"] = ((pivot[f"Agreement {last_proj_year}"] - pivot[f"Baseline {last_proj_year}"]) /
                                pivot[f"Baseline {last_proj_year}"] * 100)

print("\n=== Projected changes by bloc and MFA category ===")
print(pivot[["bloc", "mfa_category", "change_baseline_%", "change_agreement_%",
             "agreement_effect_%"]].to_string(index=False))

# Plot — indexed extraction (Observed 2022 = 100), single panel, greyscale
# Hatch distinguishes bloc; grey tone distinguishes scenario
plt.rcParams.update({"hatch.linewidth": 0.8})

fig, ax = plt.subplots(figsize=(10, 6))

obs_col = f"Observed {last_obs_year}"
bas_col = f"Baseline {last_proj_year}"
agr_col = f"Agreement {last_proj_year}"

# Build indexed values per bloc
bar_data = []  # list of (label, values_array, fill, hatch)
for bloc_name, hatch_pat in [("MERCOSUR", "///"), ("EU27", "xxx")]:
    bp = pivot[pivot["bloc"] == bloc_name].set_index("mfa_category").reindex(MFA_ORDER)
    obs_vals = bp[obs_col].values
    idx_bas = bp[bas_col].values / obs_vals * 100
    idx_agr = bp[agr_col].values / obs_vals * 100
    bar_data.append((f"{bloc_name} Baseline ({last_proj_year})", idx_bas, "#CCCCCC", hatch_pat))
    bar_data.append((f"{bloc_name} Agreement ({last_proj_year})", idx_agr, "#444444", hatch_pat))

x = np.arange(len(MFA_ORDER))
n_bars = len(bar_data)
width = 0.18
offsets = [(i - (n_bars - 1) / 2) * width for i in range(n_bars)]

for (label, vals, fill, hatch), off in zip(bar_data, offsets):
    bars = ax.bar(x + off, vals, width, color=fill, edgecolor="black",
                  hatch=hatch, linewidth=1.0, label=label)
    for bar in bars:
        bar.set_edgecolor("black")
        bar.set_linewidth(1.0)

# Reference line at 100 (Observed 2022)
ax.axhline(100, color="black", linestyle="--", linewidth=1.0,
           label=f"Observed ({last_obs_year})")

ax.set_xticks(x)
ax.set_xticklabels(MFA_ORDER, fontsize=10, rotation=20, ha="right")
ax.set_ylabel(f"Extraction index (Observed {last_obs_year} = 100)", fontsize=11)
ax.grid(True, alpha=0.15, axis="y")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=True, facecolor="white", edgecolor="grey",
          fontsize=9, loc="upper left")

plt.suptitle(f"Projected material extraction {last_obs_year}\u2013{last_proj_year}:\n"
             f"indexed to observed levels ({last_obs_year} = 100)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig09_material_impact.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: fig09_material_impact.png")

# ---------------------------------------------------------------------------
# 7. Sector-level impact estimates
# ---------------------------------------------------------------------------
print("\n[7/7] Computing sector-level impact...")

# For each sector, compute the agreement effect (difference in loading contribution)
sector_impact = []
for si, sec in enumerate(sector_names):
    for bloc_name in ["MERCOSUR", "EU27"]:
        mask_c = np.array([b == bloc_name for b in blocs])
        base_total = 0
        agree_total = 0
        for k in range(K):
            c_sum = country_loadings[mask_c, k].sum()
            s_val = sector_loadings[si, k]
            m_sum = material_loadings[:, k].sum()
            base_total += c_sum * s_val * m_sum * baseline_median[-1, k]
            agree_total += c_sum * s_val * m_sum * agreement_median[-1, k]

        pct_change = ((agree_total - base_total) / base_total * 100) if base_total > 0 else 0
        sector_impact.append({
            "sector": sec,
            "bloc": bloc_name,
            "baseline_index": base_total,
            "agreement_index": agree_total,
            "agreement_effect_%": pct_change,
        })

sector_df = pd.DataFrame(sector_impact)
sector_df.to_parquet(DATA_DIR / "projections_by_sector.parquet", index=False)
print(f"  Saved: projections_by_sector.parquet")

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("PROJECTION SUMMARY")
print(f"{'='*60}")

for bloc_name in ["MERCOSUR", "EU27"]:
    print(f"\n{bloc_name}:")
    bloc_sec = sector_df[sector_df["bloc"] == bloc_name]
    top_inc = bloc_sec.nlargest(5, "agreement_effect_%")
    print(f"  Top 5 sectors with INCREASED material extraction:")
    for _, row in top_inc.iterrows():
        print(f"    +{row['agreement_effect_%']:.1f}%  {row['sector'][:50]}")

# Save full projections
impact_df.to_parquet(DATA_DIR / "projections.parquet", index=False)
print(f"\n  Projections saved to {DATA_DIR / 'projections.parquet'}")

print("\nDone.")
