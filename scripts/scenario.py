"""
scenario.py — corrected stylised-scenario engine (JIE round-2 revision)
=======================================================================
Single source of truth for the MERCOSUR-EU scenario computation, shared by
05_sensitivity_analysis.py (tables) and 07_regenerate_all_figures.py (Figs 5-7).

Why this module exists
----------------------
The round-1 code computed the scenario effect as a percentage change of the
*log-space* reconstruction index, (a - b) / b, where a and b are sums of
log(1 + tonnes) reconstructions. Reviewer comment 1 correctly noted that a
proportional change of a log(1 + x) index is not a proportional change in
tonnes. Reproduced here: the round-1 operator (multiplying the log-loading by
1 + delta) implies a +392% change in tonnes for biomass, not +8.5%.

The corrected operator applies the shock as a *tonnes-proportional* change: a
component contributing to a cell has its tonnes scaled by (1 + delta_k), which
in log space is an additive shift of log(1 + delta_k) weighted by the
component's share of the cell's reconstruction. Effects are then computed in
tonnes space after back-transformation with expm1. This makes the reported
percentages genuine tonnage percentages (biomass +9.2%, metal ores +5.8%,
non-metallic minerals +2.0%).

Functions
---------
load_ntf()                 -> Ntf namedtuple of loadings, masks, coords
baseline_loadings(ntf)     -> quadratic-trend extrapolation to PROJ_END
tonnes_effect(...)         -> MFA-category tonnes % change (reviewer 1)
sector_effects(...)        -> sector-level tonnes % change (Table S5)
independent_mc(...)        -> ordering robustness under independent shocks (reviewer 2)
specialisation(...)        -> Balassa RCA / bloc shares, baseline vs scenario (reviewer 3)
"""
from pathlib import Path
from collections import namedtuple
import numpy as np
import xarray as xr

from config import (DATA_DIR, AGREEMENT_SHOCKS, COMP_LABELS,
                    AGREEMENT_YEAR, PHASE_IN_YEARS, PROJECTION_END)

DATA = DATA_DIR
# (MERCOSUR export boost, EU export boost) per component, from config (Table 1).
SHOCKS = {k: (v["mercosur_export_boost"], v["eu_export_boost"])
          for k, v in AGREEMENT_SHOCKS.items()}
COMP = COMP_LABELS
PHASE_IN = PHASE_IN_YEARS
PROJ_END = PROJECTION_END
MFA3 = ["Biomass", "Metal ores", "Non-metallic minerals"]

Ntf = namedtuple("Ntf", "K cl sl ml yl years blocs merc eu countries sectors "
                        "mat_subcats mfa_cats R2")


def load_ntf():
    ds = xr.open_dataset(DATA / "ntf_loadings.nc")
    blocs = ds["bloc"].values
    return Ntf(
        K=int(ds.attrs["optimal_K"]),
        cl=ds["country_loading"].values, sl=ds["sector_loading"].values,
        ml=ds["material_loading"].values, yl=ds["year_loading"].values,
        years=list(ds.coords["year"].values), blocs=blocs,
        merc=np.array([b == "MERCOSUR" for b in blocs]),
        eu=np.array([b == "EU27" for b in blocs]),
        countries=list(ds.coords["country"].values),
        sectors=list(ds.coords["sector"].values),
        mat_subcats=list(ds.coords["material_subcat"].values),
        mfa_cats=ds["mfa_category"].values,
        R2=float(ds.attrs["final_R2"]))


def baseline_loadings(ntf, proj_end=PROJ_END):
    """Quadratic-trend extrapolation of the six temporal loadings; returns the
    projected-year loading matrix and the effect-year (final) loading vector."""
    n_hist = len(ntf.years)
    t = np.arange(n_hist)
    proj_years = list(range(ntf.years[-1] + 1, proj_end))
    tp = np.arange(n_hist, n_hist + len(proj_years))
    trend = {k: np.polyfit(t, ntf.yl[:, k], 2) for k in range(ntf.K)}
    base = np.array([np.polyval(trend[k], tp) for k in range(ntf.K)]).T
    return base, base[-1, :], proj_years, trend


def avg_boost(ntf, k, mb, eb):
    """Country-loading-weighted average of the two bloc export boosts."""
    mw = ntf.cl[ntf.merc, k].sum()
    ew = ntf.cl[ntf.eu, k].sum()
    return (mb * mw + eb * ew) / (mw + ew) if (mw + ew) > 0 else 0.0


def central_delta(ntf):
    """Effective per-component tonnes shock under the central calibration."""
    return np.array([avg_boost(ntf, k, *SHOCKS[k]) for k in range(ntf.K)])


def _reconstruct(ntf, load_vec):
    """Cell-level baseline log-reconstruction X and per-component contributions."""
    X = np.einsum("ik,jk,lk,k->ijl", ntf.cl, ntf.sl, ntf.ml, load_vec)
    contrib = np.einsum("ik,jk,lk,k->ijlk", ntf.cl, ntf.sl, ntf.ml, load_vec)
    return X, contrib


def _agreement_tonnes(ntf, load_vec, delta, bloc_specific=None, phi=1.0):
    """Back-transformed cell tonnes under baseline and agreement.

    The shock is applied in tonnes space: a component's contribution to a cell
    has its tonnes scaled by (1 + delta_k * phi), i.e. an additive log shift of
    log(1 + delta_k * phi) weighted by that component's share of the cell.
    """
    X, contrib = _reconstruct(ntf, load_vec)
    Xsafe = np.where(X > 1e-9, X, 1.0)
    if bloc_specific is None:
        dX = np.einsum("ijlk,k->ijl", contrib, np.log1p(np.asarray(delta) * phi)) / Xsafe
    else:
        mbk, ebk = bloc_specific
        dX_m = np.einsum("ijlk,k->ijl", contrib, np.log1p(np.asarray(mbk) * phi)) / Xsafe
        dX_e = np.einsum("ijlk,k->ijl", contrib, np.log1p(np.asarray(ebk) * phi)) / Xsafe
        dX = np.where(ntf.merc[:, None, None], dX_m, dX_e)
    Tb = np.expm1(np.clip(X, None, 700.0))
    Ta = np.expm1(np.clip(X + dX, None, 700.0))
    return Tb, Ta


def tonnes_effect(ntf, load_vec, delta, bloc_specific=None, phi=1.0, mask=None):
    """MFA-category tonnes % change (agreement vs baseline) at the effect year.

    mask restricts the country dimension (e.g. ntf.merc) for bloc-specific bars.
    """
    Tb, Ta = _agreement_tonnes(ntf, load_vec, delta, bloc_specific, phi)
    if mask is not None:
        Tb, Ta = Tb[mask], Ta[mask]
    out = {}
    for c in MFA3:
        sel = (ntf.mfa_cats == c)
        b = Tb[:, :, sel].sum()
        a = Ta[:, :, sel].sum()
        out[c] = (a - b) / b * 100 if b > 0 else float("nan")
    return out


def sector_effects(ntf, delta, phi=1.0):
    """Sector-level tonnes % change, defined as the component-loading-weighted
    average of the per-component tonnes shocks:

        effect_j = sum_k ( sl[j,k] / sum_k' sl[j,k'] ) * delta_k

    This is bounded within [min delta, max delta] and stable for low-tonnage
    sectors, unlike a ratio of back-transformed cell reconstructions (which is
    numerically ill-conditioned where a sector's reconstructed tonnage is small).
    """
    delta = np.asarray(delta) * phi
    w = ntf.sl / ntf.sl.sum(axis=1, keepdims=True)   # loading share per component
    return (w * delta).sum(axis=1) * 100


def _ordering(eff):
    return eff["Biomass"] > eff["Metal ores"] > eff["Non-metallic minerals"]


def _biomass_leads(eff):
    return eff["Biomass"] == max(eff.values())


def independent_mc(ntf, load_vec, n=2000, mode="anchored", seed=12345):
    """Reviewer 2: vary the component shocks *independently* and report how
    often the ordering biomass > metal ores > non-metallic minerals survives.

    mode="unanchored": delta_k ~ U(0, 0.15), no reference to the tariff schedule
    mode="anchored"  : delta_k = central_k * (1 + U(-0.6, 0.6)), tariff-anchored
    """
    rng = np.random.default_rng(seed)
    base = central_delta(ntf)
    ok = leads = 0
    for _ in range(n):
        if mode == "unanchored":
            d = rng.uniform(0.0, 0.15, ntf.K)
        else:
            d = np.clip(base * (1 + rng.uniform(-0.6, 0.6, ntf.K)), 0.0, None)
        eff = tonnes_effect(ntf, load_vec, d)
        ok += _ordering(eff)
        leads += _biomass_leads(eff)
    return {"mode": mode, "n": n,
            "ordering_pct": ok / n * 100, "biomass_leads_pct": leads / n * 100}


def actual_bloc_category_tonnes(ref_year=None):
    """Observed extraction tonnes by bloc x MFA category from tensor_summary.

    Uses the actual GLORIA tonnes (not the log-space NTF reconstruction), so the
    baseline shares are physically correct (e.g. biomass ~62% of MERCOSUR).
    """
    import pandas as pd
    ts = pd.read_parquet(DATA / "tensor_summary.parquet")
    bloc_map = {"MERCOSUR": "MERCOSUR", "EU27": "EU27"}
    if ref_year is None:
        ref_year = int(ts["year"].max())
    sub = ts[ts["year"] == ref_year]
    out = {}
    for bn in bloc_map:
        for c in MFA3 + ["Fossil fuels"]:
            out[(bn, c)] = float(
                sub[(sub["bloc"] == bn) & (sub["mfa_category"] == c)]["tonnes"].sum())
    return out, ref_year


def category_boost_by_bloc(ntf, load_vec, bloc_specific):
    """Per-bloc MFA-category tonnes % effect (reconstruction-weighted average of
    the component shocks), used to perturb the observed extraction structure."""
    mbk, ebk = bloc_specific
    return {
        "MERCOSUR": tonnes_effect(ntf, load_vec, None, bloc_specific=(mbk, ebk),
                                  mask=ntf.merc),
        "EU27": tonnes_effect(ntf, load_vec, None, bloc_specific=(mbk, ebk),
                              mask=ntf.eu),
    }


def specialisation(ntf, load_vec, bloc_specific, ref_year=None):
    """Reviewer 3: specialisation indices from *observed* extraction tonnes,
    perturbed by the per-bloc scenario category effect.

    Returns, for baseline and agreement:
      within  = category share of the bloc's own extraction (%)
      between = bloc's share of both-bloc extraction of the category (%)
      rca     = Balassa revealed comparative advantage of the bloc in the category
    """
    actual, ref_year = actual_bloc_category_tonnes(ref_year)
    boost = category_boost_by_bloc(ntf, load_vec, bloc_specific)
    cats = MFA3 + ["Fossil fuels"]

    def indices(tonnes):
        bloc_tot = {bn: sum(tonnes[(bn, c)] for c in cats) for bn in ("MERCOSUR", "EU27")}
        world = sum(bloc_tot.values())
        out = {}
        for bn in ("MERCOSUR", "EU27"):
            for c in cats:
                pair = tonnes[("MERCOSUR", c)] + tonnes[("EU27", c)]
                within = tonnes[(bn, c)] / bloc_tot[bn]
                out[(bn, c)] = {"within": within * 100,
                                "between": tonnes[(bn, c)] / pair * 100 if pair > 0 else float("nan"),
                                "rca": within / (pair / world) if pair > 0 else float("nan")}
        return out

    agr = {(bn, c): actual[(bn, c)] * (1 + boost[bn].get(c, 0.0) / 100)
           for bn in ("MERCOSUR", "EU27") for c in cats}
    return {"baseline": indices(actual), "agreement": indices(agr), "ref_year": ref_year}
