"""
07_regenerate_all_figures.py — JIE-compliant regeneration of all figures
========================================================================
Produces the six main-text figures and four SI figures to comply with the
Journal of Industrial Ecology artwork guidelines:
  - 600 dpi (combination art);
  - NO titles inside the illustration (titles live in the captions);
  - sans-serif lettering (Arial/Helvetica), enlarged for legibility;
  - colour illustrations in RGB.

Standalone: reads only intermediate data (ntf_loadings.nc,
ntf_diagnostics.parquet, tensor_summary.parquet, Natural Earth shapefile).
The full GLORIA tensor is not required.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import geopandas as gpd
import scenario as S

BASE = Path(__file__).parent
DATA = BASE.parent / "data"      # repo layout: data/ at root, scripts/ alongside
FIGS = BASE.parent / "figures"   # repo layout: figures/ at root
DPI = 600

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
})

MFA_COLORS = {"Biomass": "#27ae60", "Metal ores": "#8e44ad",
              "Non-metallic minerals": "#f39c12", "Fossil fuels": "#2c3e50"}
MFA_ORDER = ["Biomass", "Metal ores", "Non-metallic minerals", "Fossil fuels"]
BLOC_COLORS = {"MERCOSUR": "#e74c3c", "EU27": "#3498db"}
COMP = {0: "Agricultural crops", 1: "Forestry & wood", 2: "Fishery & aquatic",
        3: "Non-metallic minerals", 4: "Livestock & grazing", 5: "Metal ores"}
SHOCKS = {0: (.12, .08), 1: (.06, .04), 2: (.08, .03),
          3: (.02, .02), 4: (.15, .10), 5: (.10, .05)}
AGREEMENT_YEAR, PHASE_IN, PROJ_END, N_BOOT, SEED = 2027, 5, 2035, 1000, 42

ds = xr.open_dataset(DATA / "ntf_loadings.nc")
K = int(ds.attrs["optimal_K"])
R2 = float(ds.attrs["final_R2"])
countries = list(ds.coords["country"].values)
blocs = list(ds.coords["bloc"].values)
years = list(ds.coords["year"].values)
yl = ds["year_loading"].values
cl = ds["country_loading"].values
sl = ds["sector_loading"].values
ml = ds["material_loading"].values
sectors = list(ds.coords["sector"].values)
mat_subcats = list(ds.coords["material_subcat"].values)
mfa_cats = list(ds.coords["mfa_category"].values)
clab = {f"C{k+1}": COMP[k] for k in range(K)}


def save(fig, name):
    fig.savefig(FIGS / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  saved", name)


# ===========================================================================
# Fig 2 — historical stacked area
# ===========================================================================
print("Fig 2 ...")
ts = pd.read_parquet(DATA / "tensor_summary.parquet")
fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
for bi, (bloc, label, ax) in enumerate(zip(["MERCOSUR", "EU27"],
                                           ["MERCOSUR-4", "EU-27"], axes)):
    pivot = (ts[ts["bloc"] == bloc].groupby(["year", "mfa_category"])["tonnes"]
             .sum().unstack("mfa_category")[MFA_ORDER] / 1e6)
    pivot.plot.area(ax=ax, color=[MFA_COLORS[c] for c in MFA_ORDER],
                    alpha=0.8, linewidth=0, legend=(bi == 0))
    ax.set_title(label, fontsize=19, fontweight="bold")
    ax.set_xlabel("Year", fontsize=16)
    ax.set_ylabel("Megatonnes" if bi == 0 else "", fontsize=16)
    ax.tick_params(labelsize=15)
    if bi == 0:
        ax.legend(loc="upper left", fontsize=15, frameon=True, framealpha=0.85)
    ax.grid(True, alpha=0.2)
save(fig, "fig02_historical_trends.png")

# ===========================================================================
# Fig 3 — material subcategory heatmap
# ===========================================================================
print("Fig 3 ...")
import seaborn as sns
fig, ax = plt.subplots(figsize=(11, 8))
sns.heatmap(ml, xticklabels=[f"C{k+1}\n{clab[f'C{k+1}']}" for k in range(K)],
            yticklabels=mat_subcats, cmap="YlOrRd", annot=True, fmt=".2f",
            annot_kws={"size": 12}, linewidths=0.5, ax=ax,
            cbar_kws={"label": "Loading"})
ax.set_ylabel("Material subcategory", fontsize=16)
ax.set_xlabel("Component", fontsize=16)
ax.tick_params(labelsize=13)
ax.figure.axes[-1].yaxis.label.set_size(14)
for lab, cat in zip(ax.get_yticklabels(), mfa_cats):
    lab.set_color(MFA_COLORS.get(cat, "black"))
    lab.set_fontweight("bold")
save(fig, "fig03_material_heatmap.png")

# ===========================================================================
# Fig 4 — choropleth
# ===========================================================================
print("Fig 4 ...")
world = gpd.read_file(DATA / "ne_110m_admin_0_countries.zip")
world["ISO_A3"] = world["ISO_A3"].replace({"-99": None, "-099": None})
for nm, code in {"France": "FRA", "Norway": "NOR", "Kosovo": "XKX"}.items():
    world.loc[world["NAME_EN"] == nm, "ISO_A3"] = code
sa_b, eu_b = [-82, -56, -20, 15], [-12, 34, 35, 72]
fig = plt.figure(figsize=(20, 5 * ((K + 1) // 2)))
gs = gridspec.GridSpec((K + 1) // 2, 4, wspace=0.08, hspace=0.30)
cmap = plt.cm.YlOrRd
for k in range(K):
    row, col = k // 2, (k % 2) * 2
    vals = cl[:, k]
    merged = world.merge(pd.DataFrame({"ISO_A3": countries, "loading": vals}),
                         on="ISO_A3", how="left")
    vmax = vals.max() * 1.05
    for panel, bounds, sub in [(col, sa_b, "MERCOSUR-4"), (col + 1, eu_b, "EU-27")]:
        ax = fig.add_subplot(gs[row, panel])
        world.plot(ax=ax, color="#f0f0f0", edgecolor="#cccccc", linewidth=0.3)
        merged[merged["loading"].notna()].plot(
            ax=ax, column="loading", cmap=cmap, vmin=0, vmax=vmax,
            edgecolor="#333333", linewidth=0.6)
        ax.set_xlim(bounds[0], bounds[2]); ax.set_ylim(bounds[1], bounds[3])
        ax.set_title(f"C{k+1}: {clab[f'C{k+1}']}\n{sub}",
                     fontsize=15, fontweight="bold")
        ax.axis("off")
        if sub == "EU-27":
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, vmax))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
            cbar.ax.tick_params(labelsize=13)
            cbar.set_label("Loading", fontsize=14)
save(fig, "fig04_choropleth_loadings.png")

# ===========================================================================
# Projection bootstrap (shared by Fig 5 and Fig 6)
# ===========================================================================
np.random.seed(SEED)
proj_years = list(range(years[-1] + 1, PROJ_END))
n_hist, n_proj = len(years), len(proj_years)
t, t_proj = np.arange(n_hist), np.arange(n_hist, n_hist + n_proj)
trend, resid = {}, {}
for k in range(K):
    c = np.polyfit(t, yl[:, k], 2)
    trend[k], resid[k] = c, yl[:, k] - np.polyval(c, t)
merc = np.array([b == "MERCOSUR" for b in blocs])
eu = np.array([b == "EU27" for b in blocs])
phase = np.clip((np.array(proj_years) - AGREEMENT_YEAR) / PHASE_IN, 0, 1)
pb = np.zeros((N_BOOT, n_proj, K)); pa = np.zeros((N_BOOT, n_proj, K))
for b in range(N_BOOT):
    for k in range(K):
        base = np.polyval(trend[k], t_proj) + np.random.choice(resid[k], n_proj, replace=True)
        mb, eb = SHOCKS[k]
        mw, ew = cl[merc, k].sum(), cl[eu, k].sum()
        ab = (mb * mw + eb * ew) / (mw + ew) if (mw + ew) > 0 else 0.0
        pb[b, :, k] = np.maximum(base, 0)
        pa[b, :, k] = np.maximum(base * (1 + ab * phase), 0)
b_med, b_lo, b_hi = np.median(pb, 0), np.percentile(pb, 5, 0), np.percentile(pb, 95, 0)
a_med, a_lo, a_hi = np.median(pa, 0), np.percentile(pa, 5, 0), np.percentile(pa, 95, 0)

# ===========================================================================
# Fig 5 — projection scenarios
# ===========================================================================
print("Fig 5 ...")
fig, axes = plt.subplots(2, 3, figsize=(20, 11)); axf = axes.flatten()
for k in range(K):
    ax = axf[k]
    ax.plot(years, yl[:, k], "k-", lw=3.0, label="Observed")
    ax.plot(proj_years, b_med[:, k], "--", color="#7f8c8d", lw=2.6, label="Baseline (no agreement)")
    ax.fill_between(proj_years, b_lo[:, k], b_hi[:, k], color="#7f8c8d", alpha=0.18)
    ax.plot(proj_years, a_med[:, k], "-", color="#e74c3c", lw=3.4, label="Agreement scenario")
    ax.fill_between(proj_years, a_lo[:, k], a_hi[:, k], color="#e74c3c", alpha=0.18)
    ax.axvline(AGREEMENT_YEAR, color="#2c3e50", lw=1.4, ls=":", alpha=0.7,
               label="Agreement enters force (2027)" if k == 0 else None)
    ax.set_title(f"C{k+1}: {clab[f'C{k+1}']}", fontsize=18, fontweight="bold")
    ax.tick_params(labelsize=17); ax.grid(True, alpha=0.2)
    if k >= 3:
        ax.set_xlabel("Year", fontsize=16)
    if k % 3 == 0:
        ax.set_ylabel("Temporal loading", fontsize=16)
    if k == 0:
        ax.legend(fontsize=15, frameon=True, framealpha=0.9, loc="lower left")
save(fig, "fig05_projection_scenarios.png")

# ===========================================================================
# Fig 6 — central-scenario extraction by MFA category, indexed to 2022 = 100
# Effects are in TONNES space (reviewer comment 1): the agreement path is the
# baseline tonnage trend scaled by the central tonnes-proportional effect.
# ===========================================================================
print("Fig 6 ...")
plt.rcParams.update({"hatch.linewidth": 0.8})

ntf_s = S.load_ntf()
_, eff_load_s, _, _ = S.baseline_loadings(ntf_s)
eff_uniform = S.tonnes_effect(ntf_s, eff_load_s, S.central_delta(ntf_s))  # symmetric %
cats3 = ["Biomass", "Metal ores", "Non-metallic minerals"]


def cat_tonnes_index(bloc_name):
    """Baseline-2034 and agreement-2034 category tonnes, indexed to 2022 = 100.
    Baseline is a quadratic extrapolation of the observed category tonnage; the
    agreement applies the central tonnes-proportional effect for the category."""
    idx_b, idx_a = [], []
    for c in cats3:
        s = (ts[(ts["bloc"] == bloc_name) & (ts["mfa_category"] == c)]
             .groupby("year")["tonnes"].sum().sort_index())
        yrs = s.index.values.astype(float)
        th = yrs - yrs[0]
        coef = np.polyfit(th, s.values, 1)   # linear: stable 12-year extrapolation
        obs2022 = s.values[-1]
        base2034 = np.polyval(coef, 2034 - yrs[0])
        idx_b.append(base2034 / obs2022 * 100)
        idx_a.append(base2034 * (1 + eff_uniform[c] / 100) / obs2022 * 100)
    return idx_b, idx_a


fig, ax = plt.subplots(figsize=(11, 7))
x = np.arange(len(cats3))
bar_data = []
for bloc_name, lbl, hatch in [("MERCOSUR", "MERCOSUR-4", "///"), ("EU27", "EU-27", "xxx")]:
    idx_b, idx_a = cat_tonnes_index(bloc_name)
    bar_data.append((f"{lbl} baseline 2034", idx_b, "#CCCCCC", hatch))
    bar_data.append((f"{lbl} agreement 2034", idx_a, "#444444", hatch))
width = 0.18
offs = [(i - (len(bar_data) - 1) / 2) * width for i in range(len(bar_data))]
for (lbl, vals, fill, hatch), off in zip(bar_data, offs):
    ax.bar(x + off, vals, width, color=fill, edgecolor="black", hatch=hatch,
           linewidth=1.0, label=lbl)
ax.axhline(100, color="black", ls="--", lw=1.0, label="Observed 2022")
ax.set_xticks(x); ax.set_xticklabels(cats3, fontsize=15)
ax.set_ylabel("Extraction index (observed 2022 = 100)", fontsize=15)
ax.tick_params(labelsize=14)
ax.grid(True, alpha=0.15, axis="y")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(frameon=True, facecolor="white", edgecolor="grey", fontsize=12, loc="upper left")
save(fig, "fig06_material_impact.png")

# ===========================================================================
# Fig 7 — ordinal robustness under INDEPENDENT shock variation (reviewer 2)
# The component shocks are varied independently (not by a common multiplier),
# so their relative ordering can change. The panel shows the distribution of the
# category effect across draws and reports how often the ordering survives.
# ===========================================================================
print("Fig 7 ...")
cats3 = ["Biomass", "Metal ores", "Non-metallic minerals"]
colr = {"Biomass": MFA_COLORS["Biomass"], "Metal ores": MFA_COLORS["Metal ores"],
        "Non-metallic minerals": MFA_COLORS["Non-metallic minerals"]}
cd_s = S.central_delta(ntf_s)
central = S.tonnes_effect(ntf_s, eff_load_s, cd_s)
rng = np.random.default_rng(12345)
draws = {c: [] for c in cats3}
n_draws = 2000
for _ in range(n_draws):
    d = np.clip(cd_s * (1 + rng.uniform(-0.6, 0.6, ntf_s.K)), 0.0, None)
    e = S.tonnes_effect(ntf_s, eff_load_s, d)
    for c in cats3:
        draws[c].append(e[c])
anch = S.independent_mc(ntf_s, eff_load_s, n=n_draws, mode="anchored")
unanch = S.independent_mc(ntf_s, eff_load_s, n=n_draws, mode="unanchored")

fig, ax = plt.subplots(figsize=(9, 6))
x = np.arange(len(cats3))
bp = ax.boxplot([draws[c] for c in cats3], positions=x, widths=0.5,
                patch_artist=True, showfliers=False, whis=(5, 95), zorder=1)
for patch, c in zip(bp["boxes"], cats3):
    patch.set_facecolor(colr[c]); patch.set_alpha(0.35); patch.set_edgecolor("#555555")
for med in bp["medians"]:
    med.set_color("#555555")
ax.scatter(x, [central[c] for c in cats3], s=110, color="#c0392b", zorder=3,
           label="Central illustrative scenario")
for i, c in enumerate(cats3):
    ax.annotate(f"{central[c]:+.1f}%", (x[i], central[c]), textcoords="offset points",
                xytext=(14, 0), fontsize=13, color="#c0392b", va="center")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(cats3, fontsize=15)
ax.set_ylabel("Scenario effect at 2034 vs. baseline (tonnes, %)", fontsize=15)
ax.tick_params(labelsize=14)
ax.set_title("")
ax.text(0.03, 0.03,
        f"Ordering biomass > metal ores > non-metallic minerals held in\n"
        f"{anch['ordering_pct']:.0f}% of tariff-anchored draws "
        f"(biomass leads {anch['biomass_leads_pct']:.0f}%),\n"
        f"but only {unanch['ordering_pct']:.0f}% of unanchored draws "
        f"(biomass leads {unanch['biomass_leads_pct']:.0f}%)",
        transform=ax.transAxes, fontsize=11, va="bottom", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="grey", alpha=0.9))
ax.legend(frameon=False, fontsize=13, loc="upper right")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(True, axis="y", alpha=0.2)
save(fig, "fig07_sensitivity.png")

# ===========================================================================
# Fig S1 — full component loading profiles (country, sector, material, year)
# ===========================================================================
print("Fig S1 ...")
from matplotlib.patches import Patch


def trunc(s, n):
    """Truncate a long label, marking the cut with an ellipsis."""
    return s if len(s) <= n else s[:n - 1].rstrip() + "…"


fig, axes = plt.subplots(K, 4, figsize=(22, 4 * K), squeeze=False,
                         layout="constrained")
COL_TITLES = ["Country loadings (top 15)", "Sector loadings (top 10)",
              "Material subcategory loadings", "Temporal loading"]
for j, t in enumerate(COL_TITLES):
    axes[0, j].set_title(t, fontsize=15, fontweight="bold", pad=12)
for k in range(K):
    ax = axes[k, 0]
    vals = cl[:, k]; cols = ["#e74c3c" if b == "MERCOSUR" else "#3498db" for b in blocs]
    order = np.argsort(vals)[::-1][:15]
    ax.barh(range(len(order)), vals[order], color=[cols[i] for i in order])
    ax.set_yticks(range(len(order))); ax.set_yticklabels([countries[i] for i in order], fontsize=11)
    ax.set_ylabel(f"C{k+1}: {clab[f'C{k+1}']}", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.tick_params(labelsize=11)
    if k == 0:
        ax.legend(handles=[Patch(color="#e74c3c", label="MERCOSUR-4"),
                           Patch(color="#3498db", label="EU-27")],
                  loc="lower right", fontsize=11, frameon=False)
    ax = axes[k, 1]
    vals = sl[:, k]; order = np.argsort(vals)[::-1][:10]
    ax.barh(range(len(order)), vals[order], color="#2c3e50")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([trunc(sectors[i], 35) for i in order], fontsize=10)
    ax.invert_yaxis()
    ax.tick_params(labelsize=11)
    ax = axes[k, 2]
    bar_colors = [MFA_COLORS.get(c, "#95a5a6") for c in mfa_cats]
    ax.barh(range(len(mat_subcats)), ml[:, k], color=bar_colors)
    ax.set_yticks(range(len(mat_subcats))); ax.set_yticklabels(mat_subcats, fontsize=10)
    ax.invert_yaxis()
    ax.tick_params(labelsize=11)
    ax = axes[k, 3]
    ax.plot(years, yl[:, k], "-o", color="#2c3e50", markersize=3)
    ax.tick_params(labelsize=11)
    if k == K - 1:
        ax.set_xlabel("Year", fontsize=12)
save(fig, "fig_s1_component_loadings.png")

# ===========================================================================
# Fig S2 — sector loading comparison MERCOSUR vs EU
# ===========================================================================
print("Fig S2 ...")
n_rows = (K + 2) // 3
fig, axes = plt.subplots(n_rows, 3, figsize=(20, 6 * n_rows), squeeze=False,
                         layout="constrained")
axf = axes.flatten()
for idx in range(K, n_rows * 3):
    axf[idx].set_visible(False)
mw_all = {k: cl[merc, k].mean() for k in range(K)}
ew_all = {k: cl[eu, k].mean() for k in range(K)}
for k in range(K):
    ax = axf[k]
    s_load = sl[:, k]; top = np.argsort(s_load)[::-1][:10]
    y = np.arange(len(top)); bh = 0.35
    ax.barh(y - bh / 2, s_load[top] * mw_all[k], bh, color=BLOC_COLORS["MERCOSUR"],
            label="MERCOSUR-4" if k == 0 else None, alpha=0.85)
    ax.barh(y + bh / 2, s_load[top] * ew_all[k], bh, color=BLOC_COLORS["EU27"],
            label="EU-27" if k == 0 else None, alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels([trunc(sectors[i], 40) for i in top], fontsize=10)
    ax.invert_yaxis()
    ax.set_title(f"C{k+1}: {clab[f'C{k+1}']}", fontsize=14, fontweight="bold")
    ax.tick_params(labelsize=11); ax.grid(True, alpha=0.2, axis="x")
    if k == 0:
        ax.legend(frameon=True, framealpha=0.9, fontsize=12, loc="lower right")
save(fig, "fig_s2_sector_comparison.png")

# ===========================================================================
# Fig S3 — rank selection diagnostics
# ===========================================================================
print("Fig S3 ...")
diag = pd.read_parquet(DATA / "ntf_diagnostics.parquet")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
ax.plot(diag["K"], diag["R2"], "o-", color="#2c3e50", lw=2.5, markersize=7)
ax.axvline(6, color="#c0392b", ls=":", lw=1.5)
ax.set_xlabel("Number of components (K)", fontsize=14)
ax.set_ylabel("Variance explained (R²)", fontsize=14)
ax.set_title("a) Model fit", fontsize=14, loc="left")
ax.set_xticks(list(diag["K"])); ax.tick_params(labelsize=12); ax.grid(True, alpha=0.3)
ax = axes[1]
marg = diag["R2"].diff()
ax.bar(diag["K"].iloc[1:], marg.iloc[1:], color="#2c3e50", alpha=0.75)
ax.set_xlabel("Number of components (K)", fontsize=14)
ax.set_ylabel("Marginal R² gain", fontsize=14)
ax.set_title("b) Marginal improvement", fontsize=14, loc="left")
ax.set_xticks(list(diag["K"])); ax.tick_params(labelsize=12); ax.grid(True, alpha=0.3)
save(fig, "fig_s3_rank_selection.png")

# ===========================================================================
# Fig S4 — temporal evolution by bloc
# ===========================================================================
print("Fig S4 ...")
n_rows = (K + 2) // 3
fig, axes = plt.subplots(n_rows, 3, figsize=(18, 4.5 * n_rows), sharex=True, squeeze=False)
axf = axes.flatten()
for idx in range(K, n_rows * 3):
    axf[idx].set_visible(False)
for k in range(K):
    ax = axf[k]
    for bloc_name, color, lbl in [("MERCOSUR", "#e74c3c", "MERCOSUR-4"),
                                  ("EU27", "#3498db", "EU-27")]:
        mask = np.array([b == bloc_name for b in blocs])
        w = cl[mask, k].mean()
        ax.plot(years, yl[:, k] * w, "-", color=color, lw=2.5, label=lbl)
        ax.fill_between(years, 0, yl[:, k] * w, color=color, alpha=0.1)
    ax.set_title(f"C{k+1}: {clab[f'C{k+1}']}", fontsize=15, fontweight="bold")
    ax.tick_params(labelsize=13); ax.grid(True, alpha=0.2)
    if k >= K - 3:
        ax.set_xlabel("Year", fontsize=14)
    if k % 3 == 0:
        ax.set_ylabel("Weighted year loading", fontsize=14)
    if k == 0:
        ax.legend(frameon=False, fontsize=13)
save(fig, "fig_s4_temporal_blocs.png")

print("All figures regenerated at", DPI, "dpi without in-figure titles.")
