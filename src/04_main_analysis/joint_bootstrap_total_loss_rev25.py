"""rev25 (partial-cleaning dose). Joint moving-block bootstrap of the clear-day PV model -> 95 % CI of the TOTAL annual haze loss (2024-2025 mean).
Each replicate refits beta_A and beta_S together (keeping their covariance); the hinge model re-selects its knot by AIC in
every replicate, so model-selection uncertainty is included. Losses use the same formulas as step 4 (Eqs. 7-8)."""
import sys, json, numpy as np, pandas as pd, statsmodels.api as sm
D3, OUT = sys.argv[1:3]; B = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
KWP = {"Z1": 2256.24, "Z2": 207.06, "Z3": 49.90}; KWP_TOT = sum(KWP.values()); TARIFF, EF = 4.2, 0.4857
CLEAN = pd.to_datetime(["2023-05-06", "2024-03-09", "2024-06-23", "2025-04-25", "2025-12-26", "2026-03-23", "2026-09-05"])
d = pd.read_csv(D3, index_col=0, parse_dates=True)
AOD_BG = d.loc[d.index.month.isin([11, 12, 1]), "aod_ceres"].median(); CLRW = (d.ceres_all / d.ceres_clr).clip(0, 1)
COMMISSION = pd.Timestamp("2023-03-03")   # step-9 fix: new modules start clean (dose reset), as in step 3 / Table 3
PF = 2.0   # rev25: partial-cleaning dose, complete cleaning at PF mm
wash = d.index.isin(CLEAN + pd.Timedelta(days=1)) | (d.index == COMMISSION); keep = 1 - np.minimum(1.0, d.e5_tp.fillna(0).values / PF)
_x = d.pm25.values / 1000.0; _o = np.full(len(d), np.nan); _D = 0.0
for _i in range(len(d)):
    _D = 0.0 if wash[_i] else _D * keep[_i]
    if not np.isnan(_x[_i]): _D += _x[_i]; _o[_i] = _D
d["dose"] = _o
rows = []
for z in KWP:
    t = pd.DataFrame({"ratio": d[f"Y_{z}"] / d.e5_ssrd, "aod": d.aod_ceres, "t2m": d.e5_t2m, "soil": d.dose,
                      "clear": (d.e5_tcc_day < 30) & (d.ceres_all / d.ceres_clr > 0.85) & (d.e5_tp < 1),
                      "yr": (d.index - pd.Timestamp("2023-03-03")).days / 365.25, "zone": z}, index=d.index)
    med = t.ratio.median(); rows.append(t[t.clear & t.ratio.between(0.4 * med, 1.3 * med)])
PV = pd.concat(rows).drop(columns="clear").dropna()
def X(df, aod):
    x = pd.get_dummies(df.zone).astype(float)
    for z in KWP: x[f"tr_{z}"] = (df.zone == z) * df.yr
    x["aod"] = aod; x[["t2m", "soil"]] = df[["t2m", "soil"]]; return x
# annual exposure sums (2024-2025 mean) so that loss = -beta * S
act = sum(d[f"Y_{z}"] * k for z, k in KWP.items()); ok = act.notna() & d.index.year.isin([2024, 2025])
def annual(s): return float(s[ok].groupby(s[ok].index.year).sum().mean() / 1000)          # MWh a year
S_atm_lin = annual((d.aod_ceres - AOD_BG).clip(lower=0) * d.e5_ssrd * CLRW * KWP_TOT)
KNOTS = np.round(np.arange(0.10, 0.61, 0.05), 2)
S_atm_h = {k: annual(((d.aod_ceres - k).clip(lower=0) - max(AOD_BG - k, 0)).clip(lower=0) * d.e5_ssrd * CLRW * KWP_TOT) for k in KNOTS}
S_dep = annual(d.dose * d.e5_ssrd * KWP_TOT)
E_act = annual(act)
def losses(df):
    m = sm.OLS(df.ratio, X(df, df.aod)).fit()
    lin = dict(atm=-m.params.aod * S_atm_lin, dep=-m.params.soil * S_dep)
    best = None
    for k in KNOTS:
        mh = sm.OLS(df.ratio, X(df, (df.aod - k).clip(lower=0))).fit()
        if best is None or mh.aic < best[0]: best = (mh.aic, k, mh)
    _, k, mh = best; hin = dict(atm=-mh.params.aod * S_atm_h[k], dep=-mh.params.soil * S_dep, knot=k)
    for v in (lin, hin): v["tot"] = v["atm"] + v["dep"]
    return lin, hin
point_lin, point_hin = losses(PV)
blk = (PV.index - PV.index.min()).days // 14; ub = np.unique(blk); rng = np.random.default_rng(42); rep = []
groups = {k: PV[blk == k] for k in ub}
for b in range(B):
    PVb = pd.concat([groups[k] for k in rng.choice(ub, len(ub), replace=True)])
    l, h = losses(PVb); rep.append(dict(lin_atm=l["atm"], lin_dep=l["dep"], lin_tot=l["tot"], hin_atm=h["atm"], hin_dep=h["dep"], hin_tot=h["tot"], knot=h["knot"]))
BT = pd.DataFrame(rep); BT.to_csv(f"{OUT}/joint_bootstrap_replicates.csv", index=False)
q = lambda s: [round(float(s.quantile(0.025)), 1), round(float(s.quantile(0.975)), 1)]
pooled = pd.concat([BT.lin_tot, BT.hin_tot])                                    # model-averaged (equal weight) interval
R = dict(B=B, point_linear={k: round(v, 1) for k, v in point_lin.items()}, point_hinge={k: (round(v, 1) if k != "knot" else v) for k, v in point_hin.items()},
         CI95_linear=dict(atm=q(BT.lin_atm), dep=q(BT.lin_dep), tot=q(BT.lin_tot)),
         CI95_hinge=dict(atm=q(BT.hin_atm), dep=q(BT.hin_dep), tot=q(BT.hin_tot)),
         CI95_pooled_tot=q(pooled), corr_atm_dep_linear=round(float(BT.lin_atm.corr(BT.lin_dep)), 2),
         knot_selected_share={str(k): round(float(v), 3) for k, v in BT.knot.value_counts(normalize=True).sort_index().items()},
         E_actual_MWh=round(E_act, 1))
for lab, s in (("linear", BT.lin_tot), ("hinge", BT.hin_tot), ("pooled", pooled)):
    lo, hi = s.quantile(0.025), s.quantile(0.975)
    R[f"pct_of_potential_{lab}"] = [round(100 * lo / (E_act + lo), 1), round(100 * hi / (E_act + hi), 1)]
    R[f"THB_million_{lab}"] = [round(lo * TARIFF / 1000, 2), round(hi * TARIFF / 1000, 2)]
    R[f"tCO2_{lab}"] = [round(lo * EF), round(hi * EF)]
R["atm_fraction_CI95_linear"] = q(100 * BT.lin_atm / BT.lin_tot)
json.dump(R, open(f"{OUT}/joint_bootstrap_total_loss.json", "w"), indent=1, default=float); print(json.dumps(R, indent=1, default=float))
