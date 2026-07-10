"""
05_sensitivity_analysis.py — Stylised sensitivity analysis (JIE round-2 revision)
=================================================================================
Prints the tables underlying Section 4.5 and the Supplementary Information,
using the corrected scenario engine in scenario.py.

Round-2 reviewer comments addressed here:
  1. Effects are computed in TONNES space (tonnes-proportional shock,
     back-transformed) instead of as a ratio of log-space indices.
  2. Ordinal robustness is tested by varying the component shocks INDEPENDENTLY
     (anchored and unanchored), not by a common multiplier on the whole vector.
  3. A specialisation measure is defined explicitly and evaluated baseline vs
     scenario, distinguishing the within-bloc compositional shift (which moves)
     from the between-bloc asymmetry (invariant to a common boost by construction).

Retained from round 1: NTF-vs-naive-MFA-subcategory comparison and the
uniformly-screened rank diagnostics.

Inputs:  data/ntf_loadings.nc, data/ntf_diagnostics.parquet, data/tensor_summary.parquet
Outputs: printed tables only. Figures 5-7 are produced by 07_regenerate_all_figures.py.
"""
import numpy as np
import pandas as pd
import scenario as S

ntf = S.load_ntf()
base_load, eff_load, proj_years, trend = S.baseline_loadings(ntf)
cd = S.central_delta(ntf)
mbk = np.array([S.SHOCKS[k][0] for k in range(ntf.K)])
ebk = np.array([S.SHOCKS[k][1] for k in range(ntf.K)])
EFFECT_YEAR = proj_years[-1]

print(f"NTF: K={ntf.K}, R2={ntf.R2:.3f}; effect year = {EFFECT_YEAR}")
print("Central per-component tonnes shock (C1..C6):", np.round(cd, 4))

# --- Reviewer 1: category effects in TONNES space (central scenario) ---
central = S.tonnes_effect(ntf, eff_load, cd)
print("\n=== Central-scenario MFA-category effect at 2034 (tonnes space) ===")
for c in S.MFA3:
    print(f"  {c:24s} {central[c]:+.2f}%")
print("  (round-1 log-space index reported +8.5/+5.8/+2.0; the tonnes-proportional")
print("   operator now yields genuine tonnage percentages)")

# --- Reviewer 2: ordinal robustness under INDEPENDENT shock variation ---
print("\n=== Ordinal robustness: independent component shocks (n=2000) ===")
for mode in ("anchored", "unanchored"):
    r = S.independent_mc(ntf, eff_load, n=2000, mode=mode)
    tag = {"anchored": "tariff-anchored +/-60%",
           "unanchored": "unanchored U(0,0.15)"}[mode]
    print(f"  {tag:24s}  ordering held {r['ordering_pct']:.1f}%  "
          f"biomass leads {r['biomass_leads_pct']:.1f}%")

# per-category effect distribution across anchored draws (Table S7 ranges)
rng = np.random.default_rng(12345)
dist = {c: [] for c in S.MFA3}
for _ in range(2000):
    d = np.clip(cd * (1 + rng.uniform(-0.6, 0.6, ntf.K)), 0.0, None)
    e = S.tonnes_effect(ntf, eff_load, d)
    for c in S.MFA3:
        dist[c].append(e[c])
print("\n=== Effect distribution over anchored draws (central | 5th-95th | min-max) ===")
for c in S.MFA3:
    v = np.array(dist[c])
    print(f"  {c:24s} {central[c]:+.2f}% | {np.percentile(v,5):+.1f} to "
          f"{np.percentile(v,95):+.1f} | {v.min():+.1f} to {v.max():+.1f}")

# --- Reviewer 3: specialisation, observed tonnes, baseline vs scenario ---
print("\n=== Specialisation (observed 2022 extraction shares) ===")
# Uniform (headline) scenario: within-bloc composition shifts; between-bloc invariant.
actual, ry = S.actual_bloc_category_tonnes()
cats = S.MFA3 + ["Fossil fuels"]
uboost = {c: central.get(c, 0.0) for c in cats}
uboost["Fossil fuels"] = 0.0
print(f"  Reference year: {ry}")
print("  -- headline uniform scenario (within-bloc category shares) --")
for bl in ("MERCOSUR", "EU27"):
    b0 = {c: actual[(bl, c)] for c in cats}
    t0 = sum(b0.values())
    b1 = {c: actual[(bl, c)] * (1 + uboost[c] / 100) for c in cats}
    t1 = sum(b1.values())
    print(f"     {bl:8s} biomass {b0['Biomass']/t0*100:5.2f}->{b1['Biomass']/t1*100:5.2f}%"
          f"  metal {b0['Metal ores']/t0*100:5.2f}->{b1['Metal ores']/t1*100:5.2f}%"
          f"  non-met {b0['Non-metallic minerals']/t0*100:5.2f}->{b1['Non-metallic minerals']/t1*100:5.2f}%")
print("  (between-bloc shares are invariant to a common proportional boost)")

# Bloc-specific variant (Table S9): between-bloc asymmetry moves.
sp = S.specialisation(ntf, eff_load, bloc_specific=(mbk, ebk))
print("  -- bloc-specific variant (Table S9): MERCOSUR indices --")
for c in S.MFA3:
    b = sp["baseline"][("MERCOSUR", c)]
    a = sp["agreement"][("MERCOSUR", c)]
    print(f"     {c:22s} within {b['within']:5.2f}->{a['within']:5.2f}%  "
          f"between {b['between']:5.2f}->{a['between']:5.2f}%  RCA {b['rca']:.3f}->{a['rca']:.3f}")
cbe = S.category_boost_by_bloc(ntf, eff_load, (mbk, ebk))
print("     per-bloc category effect: MERC",
      {c: round(cbe['MERCOSUR'][c], 1) for c in S.MFA3},
      "| EU", {c: round(cbe['EU27'][c], 1) for c in S.MFA3})

# --- Table S5: sector-level effect (loading-weighted, tonnes) ---
print("\n=== Table S5: sector-level tonnes effect (top 20) ===")
se = S.sector_effects(ntf, cd)
order = np.argsort(se)[::-1][:20]
for rank, i in enumerate(order, 1):
    print(f"  {rank:2d}. {se[i]:+6.2f}%  {ntf.sectors[i]}")

# --- Retained: NTF vs naive MFA-subcategory (reviewer comment 4, round 1) ---
print("\n=== NTF material-subcategory loadings by component (near-diagonal check) ===")
df = pd.DataFrame(ntf.ml, index=ntf.mat_subcats,
                  columns=[f"C{k+1}" for k in range(ntf.K)])
pd.set_option("display.width", 200, "display.max_columns", 20)
print(df.round(3).to_string())

c5 = 4
print("\n=== C5 (livestock) cross-dimensional coupling ===")
top_sec = np.argsort(ntf.sl[:, c5])[::-1][:5]
print("  Top C5 sectors:", [(ntf.sectors[i][:35], round(ntf.sl[i, c5], 3)) for i in top_sec])
top_mat = np.argsort(ntf.ml[:, c5])[::-1][:3]
print("  Top C5 materials:", [(ntf.mat_subcats[i], round(ntf.ml[i, c5], 3)) for i in top_mat])
top_cty = np.argsort(ntf.cl[:, c5])[::-1][:3]
print("  Top C5 countries:", [(ntf.countries[i], round(ntf.cl[i, c5], 3)) for i in top_cty])
c6 = 5
top_cty6 = np.argsort(ntf.cl[:, c6])[::-1][:3]
print("  Top C6 countries:", [(ntf.countries[i], round(ntf.cl[i, c6], 3)) for i in top_cty6])

# --- Retained: uniformly-screened rank diagnostics (reviewer comment 3, round 1) ---
diag = pd.read_parquet(S.DATA / "ntf_diagnostics.parquet")
diag["marginal_R2_gain"] = diag["R2"].diff()
print("\n=== Rank diagnostics (uniform 3 initialisations per K) ===")
print(diag.round(3).to_string(index=False))

print("\nDone. Figures 5-7 are generated by 07_regenerate_all_figures.py.")
