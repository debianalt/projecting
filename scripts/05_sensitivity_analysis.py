"""
05_sensitivity_analysis.py — Stylised scenario sensitivity analysis
===================================================================
Maps the sensitivity of the trade-agreement scenario effect to the assumed
shock magnitude and phase-in length, addressing the conditional nature of
the projection: the scenario shocks are illustrative parameters, not
estimated trade-policy responses, so the magnitude of any effect is by
construction proportional to the assumed shock.

The script sweeps the full shock vector over scale factors 0.5-1.5 and the
phase-in window over 3-10 years (20 combinations), reports the resulting
envelope of the MFA-category effect, checks ordinal robustness, and produces
fig07_sensitivity.png. It also prints a comparison of the NTF components
against a naive reading of the material classification.

Inputs:  data/ntf_loadings.nc
Outputs: figures/fig07_sensitivity.png
         printed sensitivity envelope and NTF-vs-classification tables

Runs from the intermediate data alone; the full GLORIA tensor is not needed.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from config import (
    DATA_DIR, FIG_DIR, COMP_LABELS, AGREEMENT_SHOCKS, PHASE_IN_YEARS,
    AGREEMENT_YEAR, PROJECTION_END, YEAR_RANGE,
)

MFA_ORDER = ["Biomass", "Metal ores", "Non-metallic minerals"]

loadings = xr.open_dataset(DATA_DIR / "ntf_loadings.nc")
K = int(loadings.attrs["optimal_K"])
years = list(loadings.coords["year"].values)
blocs = list(loadings.coords["bloc"].values)
yl = loadings["year_loading"].values
cl = loadings["country_loading"].values
sl = loadings["sector_loading"].values
ml = loadings["material_loading"].values
mat_subcats = list(loadings.coords["material_subcat"].values)
mfa_cats = list(loadings.coords["mfa_category"].values)
sectors = list(loadings.coords["sector"].values)
countries = list(loadings.coords["country"].values)

proj_years = list(range(YEAR_RANGE[1] + 1, PROJECTION_END))
n_hist, n_proj = len(years), len(proj_years)
t = np.arange(n_hist)
tp = np.arange(n_hist, n_hist + n_proj)
trend = {k: np.polyfit(t, yl[:, k], 2) for k in range(K)}
base_load = np.array([np.polyval(trend[k], tp) for k in range(K)]).T

merc = np.array([b == "MERCOSUR" for b in blocs])
eu = np.array([b == "EU27" for b in blocs])


def avg_boost(k, scale=1.0, equal=False):
    if equal:
        mb, eb = 0.08, 0.05
    else:
        mb = AGREEMENT_SHOCKS[k]["mercosur_export_boost"]
        eb = AGREEMENT_SHOCKS[k]["eu_export_boost"]
    mb *= scale
    eb *= scale
    mw, ew = cl[merc, k].sum(), cl[eu, k].sum()
    tw = mw + ew
    return (mb * mw + eb * ew) / tw if tw > 0 else 0.0


def mfa_effects(scale=1.0, phase=PHASE_IN_YEARS, equal=False):
    ph = np.clip((np.array(proj_years) - AGREEMENT_YEAR) / phase, 0, 1)
    agl = base_load.copy()
    for k in range(K):
        agl[:, k] = base_load[:, k] * (1 + avg_boost(k, scale, equal) * ph)
    by, ay = base_load[-1, :], agl[-1, :]
    cb, ca = {}, {}
    for mi, cat in enumerate(mfa_cats):
        b = sum(cl[:, k].sum() * sl[:, k].sum() * ml[mi, k] * by[k] for k in range(K))
        a = sum(cl[:, k].sum() * sl[:, k].sum() * ml[mi, k] * ay[k] for k in range(K))
        cb[cat] = cb.get(cat, 0) + b
        ca[cat] = ca.get(cat, 0) + a
    return {c: (ca[c] - cb[c]) / cb[c] * 100 for c in cb if cb[c] > 0}


scales = [0.5, 0.75, 1.0, 1.25, 1.5]
phases = [3, 5, 7, 10]
central = mfa_effects()
print("Central scenario effect (scale=1, phase=5):")
for c in MFA_ORDER:
    print(f"  {c:24s} {central[c]:+.2f}%")

env = {c: [] for c in MFA_ORDER}
ordering_ok = 0
for s in scales:
    for ph in phases:
        e = mfa_effects(s, ph)
        for c in MFA_ORDER:
            env[c].append(e[c])
        if e["Biomass"] > e["Metal ores"] > e["Non-metallic minerals"]:
            ordering_ok += 1
print(f"\nEnvelope across {len(scales)*len(phases)} combinations:")
for c in MFA_ORDER:
    print(f"  {c:24s} {min(env[c]):+.2f}% to {max(env[c]):+.2f}%")
print(f"Ordinal ranking biomass>metal>non-metallic preserved in "
      f"{ordering_ok}/{len(scales)*len(phases)} combinations")

eq = mfa_effects(equal=True)
print("\nEqual-shock alternative (isolates the assumed ordering):")
for c in MFA_ORDER:
    print(f"  {c:24s} {eq[c]:+.2f}%")

# Figure 7 — sensitivity envelope
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(MFA_ORDER))
lows = [min(env[c]) for c in MFA_ORDER]
highs = [max(env[c]) for c in MFA_ORDER]
cents = [central[c] for c in MFA_ORDER]
for i in range(len(MFA_ORDER)):
    ax.plot([x[i], x[i]], [lows[i], highs[i]], color="#555555", lw=2, zorder=1)
    ax.plot([x[i] - 0.08, x[i] + 0.08], [lows[i], lows[i]], color="#555555", lw=2)
    ax.plot([x[i] - 0.08, x[i] + 0.08], [highs[i], highs[i]], color="#555555", lw=2)
ax.scatter(x, cents, s=90, color="#c0392b", zorder=3, label="Central illustrative scenario")
for i in range(len(MFA_ORDER)):
    ax.annotate(f"{cents[i]:+.1f}%", (x[i], cents[i]), textcoords="offset points",
                xytext=(10, 0), fontsize=9, color="#c0392b", va="center")
    ax.annotate(f"{highs[i]:+.1f}", (x[i], highs[i]), textcoords="offset points",
                xytext=(0, 6), fontsize=8, color="#555555", ha="center")
    ax.annotate(f"{lows[i]:+.1f}", (x[i], lows[i]), textcoords="offset points",
                xytext=(0, -14), fontsize=8, color="#555555", ha="center")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(MFA_ORDER, fontsize=10)
ax.set_ylabel("Scenario effect at 2034 vs. baseline trend (%)", fontsize=10)
ax.set_title("Sensitivity of the MFA-category scenario effect to assumed shock magnitude\n"
             "(range: shock scale 0.5-1.5 x phase-in 3-10 years)", fontsize=10.5)
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, axis="y", alpha=0.2)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig07_sensitivity.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"\nSaved: {FIG_DIR / 'fig07_sensitivity.png'}")

# NTF vs material classification (near-diagonal check)
print("\nNTF material-subcategory loadings (peak component per subcategory):")
for mi, sc in enumerate(mat_subcats):
    k = int(np.argmax(ml[mi]))
    print(f"  {sc:22s} peak C{k+1} ({ml[mi, k]:.3f})"
          + ("  [near-zero: not represented]" if ml[mi].max() < 1e-3 else ""))
