"""
06_regenerate_main_figures.py — Large-font versions of Figs 2, 4, 5
===================================================================
Regenerates the three main-text figures with enlarged, legible fonts
(interior labels, axis ticks, scale legends). Standalone: reads only the
intermediate data in data/ (ntf_loadings.nc, tensor_summary.parquet, the
Natural Earth shapefile) — the full GLORIA tensor is not required.

Outputs (JIE figure names) to figures/:
  fig02_historical_trends.png    stacked-area extraction by MFA category
  fig04_choropleth_loadings.png  choropleth of NTF country loadings
  fig05_projection_scenarios.png baseline vs agreement temporal loadings
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import geopandas as gpd

from config import (
    DATA_DIR, FIG_DIR, COMP_LABELS, BLOC_COLORS, MFA_COLORS, MFA_ORDER,
    AGREEMENT_SHOCKS, AGREEMENT_YEAR, PHASE_IN_YEARS, PROJECTION_END,
    N_BOOTSTRAP, RANDOM_SEED,
)

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 17, "axes.labelsize": 16,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 15,
    "figure.titlesize": 22,
})

loadings = xr.open_dataset(DATA_DIR / "ntf_loadings.nc")
K = int(loadings.attrs["optimal_K"])
R2 = float(loadings.attrs["final_R2"])
country_codes = list(loadings.coords["country"].values)
blocs = list(loadings.coords["bloc"].values)
years = list(loadings.coords["year"].values)
yl = loadings["year_loading"].values
cl = loadings["country_loading"].values
clabel = {f"C{k+1}": COMP_LABELS[k] for k in range(K)}

# --- Fig 2: historical stacked area ---
print("Fig 2 ...")
ts = pd.read_parquet(DATA_DIR / "tensor_summary.parquet")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for bi, (bloc, label, ax) in enumerate(zip(["MERCOSUR", "EU27"],
                                           ["MERCOSUR-4", "EU-27"], axes)):
    pivot = (ts[ts["bloc"] == bloc].groupby(["year", "mfa_category"])["tonnes"]
             .sum().unstack("mfa_category")[MFA_ORDER] / 1e6)
    pivot.plot.area(ax=ax, color=[MFA_COLORS[c] for c in MFA_ORDER],
                    alpha=0.8, linewidth=0)
    ax.set_title(label, fontsize=19, fontweight="bold")
    ax.set_xlabel("Year", fontsize=16)
    ax.set_ylabel("Megatonnes" if bi == 0 else "", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.legend(loc="upper left", fontsize=15, frameon=True, framealpha=0.85)
    ax.grid(True, alpha=0.2)
plt.suptitle("Material extraction by MFA category (1990-2022)",
             fontsize=22, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig02_historical_trends.png", dpi=300, bbox_inches="tight")
plt.close()

# --- Fig 4: choropleth ---
print("Fig 4 ...")
world = gpd.read_file(DATA_DIR / "ne_110m_admin_0_countries.zip")
world["ISO_A3"] = world["ISO_A3"].replace({"-99": None, "-099": None})
for nm, code in {"France": "FRA", "Norway": "NOR", "Kosovo": "XKX"}.items():
    world.loc[world["NAME_EN"] == nm, "ISO_A3"] = code
sa_bounds, eu_bounds = [-82, -56, -20, 15], [-12, 34, 35, 72]
fig = plt.figure(figsize=(20, 5 * ((K + 1) // 2)))
gs = gridspec.GridSpec((K + 1) // 2, 4, wspace=0.08, hspace=0.30)
cmap = plt.cm.YlOrRd
for k in range(K):
    row, col = k // 2, (k % 2) * 2
    vals = cl[:, k]
    merged = world.merge(pd.DataFrame({"ISO_A3": country_codes, "loading": vals}),
                         on="ISO_A3", how="left")
    vmax = vals.max() * 1.05
    for panel, bounds, sub in [(col, sa_bounds, "MERCOSUR"),
                               (col + 1, eu_bounds, "EU-27")]:
        ax = fig.add_subplot(gs[row, panel])
        world.plot(ax=ax, color="#f0f0f0", edgecolor="#cccccc", linewidth=0.3)
        merged[merged["loading"].notna()].plot(
            ax=ax, column="loading", cmap=cmap, vmin=0, vmax=vmax,
            edgecolor="#333333", linewidth=0.6)
        ax.set_xlim(bounds[0], bounds[2]); ax.set_ylim(bounds[1], bounds[3])
        ax.set_title(f"C{k+1}: {clabel[f'C{k+1}']}\n{sub}",
                     fontsize=15, fontweight="bold")
        ax.axis("off")
        if sub == "EU-27":
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, vmax))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
            cbar.ax.tick_params(labelsize=13)
            cbar.set_label("Loading", fontsize=13)
plt.suptitle(f"NTF country loadings (K={K}, R2={R2:.2f})",
             fontsize=22, fontweight="bold", y=1.005)
plt.savefig(FIG_DIR / "fig04_choropleth_loadings.png", dpi=300, bbox_inches="tight")
plt.close()

# --- Fig 5: projection scenarios ---
print("Fig 5 ...")
np.random.seed(RANDOM_SEED)
proj_years = list(range(years[-1] + 1, PROJECTION_END))
n_hist, n_proj = len(years), len(proj_years)
t, t_proj = np.arange(n_hist), np.arange(n_hist, n_hist + n_proj)
trend, resid = {}, {}
for k in range(K):
    c = np.polyfit(t, yl[:, k], 2)
    trend[k], resid[k] = c, yl[:, k] - np.polyval(c, t)
merc = np.array([b == "MERCOSUR" for b in blocs])
eu = np.array([b == "EU27" for b in blocs])
phase = np.clip((np.array(proj_years) - AGREEMENT_YEAR) / PHASE_IN_YEARS, 0.0, 1.0)
pb = np.zeros((N_BOOTSTRAP, n_proj, K)); pa = np.zeros((N_BOOTSTRAP, n_proj, K))
for b in range(N_BOOTSTRAP):
    for k in range(K):
        base = np.polyval(trend[k], t_proj) + np.random.choice(resid[k], n_proj, replace=True)
        mb, eb = AGREEMENT_SHOCKS[k]["mercosur_export_boost"], AGREEMENT_SHOCKS[k]["eu_export_boost"]
        mw, ew = cl[merc, k].sum(), cl[eu, k].sum()
        ab = (mb * mw + eb * ew) / (mw + ew) if (mw + ew) > 0 else 0.0
        pb[b, :, k] = np.maximum(base, 0)
        pa[b, :, k] = np.maximum(base * (1 + ab * phase), 0)
b_med, b_lo, b_hi = np.median(pb, 0), np.percentile(pb, 5, 0), np.percentile(pb, 95, 0)
a_med, a_lo, a_hi = np.median(pa, 0), np.percentile(pa, 5, 0), np.percentile(pa, 95, 0)
fig, axes = plt.subplots(2, 3, figsize=(20, 11)); axf = axes.flatten()
for k in range(K):
    ax = axf[k]
    ax.plot(years, yl[:, k], "k-", lw=2.5, label="Observed")
    ax.plot(proj_years, b_med[:, k], "--", color="#7f8c8d", lw=2.2, label="Baseline (no agreement)")
    ax.fill_between(proj_years, b_lo[:, k], b_hi[:, k], color="#7f8c8d", alpha=0.18)
    ax.plot(proj_years, a_med[:, k], "-", color="#e74c3c", lw=2.8, label="Agreement scenario")
    ax.fill_between(proj_years, a_lo[:, k], a_hi[:, k], color="#e74c3c", alpha=0.18)
    ax.axvline(AGREEMENT_YEAR, color="#2c3e50", lw=1.2, ls=":", alpha=0.7)
    ax.annotate("Agreement\nenters force", xy=(AGREEMENT_YEAR, ax.get_ylim()[1] * 0.93),
                fontsize=12, ha="center", color="#2c3e50", fontweight="bold")
    ax.set_title(f"C{k+1}: {clabel[f'C{k+1}']}", fontsize=18, fontweight="bold")
    ax.tick_params(labelsize=14); ax.grid(True, alpha=0.2)
    if k >= 3:
        ax.set_xlabel("Year", fontsize=16)
    if k == 0:
        ax.legend(fontsize=14, frameon=True, framealpha=0.9, loc="upper left")
plt.suptitle("Projected NTF temporal loadings: baseline vs. agreement scenario\n"
             "(shaded bands: 90% trend-only bootstrap CI, n=1,000 - not agreement-effect uncertainty)",
             fontsize=20, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(FIG_DIR / "fig05_projection_scenarios.png", dpi=300, bbox_inches="tight")
plt.close()
print("Done.")
