"""Step 6b – analyses that use the data added on 2026-10-20.
N1 attribution with hourly-wind trajectories (h850), 6-hourly sub-sample (h850_6h) and boundary-layer proxy (blproxy),
   compared with the original 6-hourly p850 set-up on the SAME sample (Jan-May 2023-2025)
N2 five-season (2021-2025) exposure climatology from hourly trajectories
N3 CAMS EAC4 aerosol speciation: carbonaceous (OM+BC) share of the haze-season AOD excess vs the model's fire share
N4 MODIS MAIAC 1-km AOD vs CERES 1-deg AOD at the site (monthly)"""
import sys, glob, json, itertools, math, warnings, h5py, tifffile, numpy as np, pandas as pd, statsmodels.api as sm
warnings.filterwarnings("ignore")
D3, TFE_OLD, TFE_NEW, EAC4, MAIAC, S5, OUT = sys.argv[1:8]
KWP = {"Z1": 2256.24, "Z2": 207.06, "Z3": 49.90}; KWP_TOT = sum(KWP.values())
CLEAN = pd.to_datetime(["2023-05-06", "2024-03-09", "2024-06-23", "2025-04-25", "2025-12-26", "2026-03-23", "2026-09-05"])
C = ["THA", "MMR", "LAO", "OTH"]; R = {}; rng = np.random.default_rng(42)
LAT0, LON0 = 19.905, 99.83
d = pd.read_csv(D3, index_col=0, parse_dates=True)
AOD_BG = d.loc[d.index.month.isin([11, 12, 1]), "aod_ceres"].median(); CLRW = (d.ceres_all / d.ceres_clr).clip(0, 1)
COMMISSION = pd.Timestamp("2023-03-03"); PF = 2.0
def soil_dose(pm, tp, thr=PF):
    wash = pm.index.isin(CLEAN + pd.Timedelta(days=1)) | (pm.index == COMMISSION)
    keep = 1 - np.minimum(1.0, tp.reindex(pm.index).fillna(0).values / thr); x = pm.values / 1000.0
    out = np.full(len(pm), np.nan); D = 0.0
    for i in range(len(pm)):
        D = 0.0 if wash[i] else D * keep[i]
        if not np.isnan(x[i]): D += x[i]; out[i] = D
    return pd.Series(out, index=pm.index)
d["dose"] = soil_dose(d.pm25, d.e5_tp)
rows = []
for z in KWP:
    t = pd.DataFrame({"ratio": d[f"Y_{z}"] / d.e5_ssrd, "aod": d.aod_ceres, "t2m": d.e5_t2m, "soil": d.dose,
                      "clear": (d.e5_tcc_day < 30) & (d.ceres_all / d.ceres_clr > 0.85) & (d.e5_tp < 1),
                      "yr": (d.index - pd.Timestamp("2023-03-03")).days / 365.25, "zone": z}, index=d.index)
    med = t.ratio.median(); rows.append(t[t.clear & t.ratio.between(0.4 * med, 1.3 * med)])
PV = pd.concat(rows).drop(columns="clear").dropna()
X = pd.get_dummies(PV.zone).astype(float)
for z in KWP: X[f"tr_{z}"] = (PV.zone == z) * PV.yr
X[["aod", "t2m", "soil"]] = PV[["aod", "t2m", "soil"]]
m0 = sm.OLS(PV.ratio, X).fit(); B_AOD, B_SOIL = m0.params.aod, m0.params.soil

MET = ["lblh", "inv850_07", "e5_rh", "e5_ws", "e5_tp", "ws850", "e5_tcwv", "s1", "c1"]
def haze_frame(tfe, years, cs=C):
    H = d.drop(columns=[f"tfe_{c}" for c in C]).join(tfe[[f"tfe_{c}" for c in cs]], how="inner")
    H = H[H.index.month.isin([1, 2, 3, 4, 5]) & H.index.year.isin(years)].copy()
    H["lblh"] = np.log(H.e5_blh_day); doy = H.index.dayofyear
    H["s1"], H["c1"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    fire = []
    for c in cs:
        for L in (0, 1): H[f"lt_{c}_{L}"] = np.log1p(H[f"tfe_{c}"].shift(L)); fire.append(f"lt_{c}_{L}")
    return H.dropna(subset=fire + MET + ["pm25", "aod_ceres"]), fire

def engine(H, fire, cs=C, atm_only=False):
    Xh = sm.add_constant(H[fire + MET]); mA = sm.OLS(H.aod_ceres, Xh).fit(); mP = sm.OLS(np.log(H.pm25), Xh).fit()
    bA, bP = mA.predict(Xh), mP.predict(Xh); e = d.reindex(H.index)
    def v(coal):
        if not coal: return 0.0, 0.0
        Z = H[fire + MET].copy()
        for cc in [x for x in fire if x.split("_")[1] in coal]: Z[cc] = np.minimum(Z[cc], H[cc].quantile(0.10))
        Zc = sm.add_constant(Z, has_constant="add")
        dA = np.minimum((bA - mA.predict(Zc)).clip(lower=0), (H.aod_ceres - AOD_BG).clip(lower=0))
        atm = float((-B_AOD * dA * e.e5_ssrd * CLRW.reindex(H.index).values * KWP_TOT / 1000 * (H.index >= COMMISSION)).sum())
        if atm_only: return atm, 0.0
        dP = H.pm25 * (1 - np.exp(-(bP - mP.predict(Zc)).clip(lower=0)))
        pm_c = d.pm25.copy(); pm_c.loc[H.index] = pm_c.loc[H.index] - dP
        dep = -B_SOIL * (d.dose - soil_dose(pm_c, d.e5_tp)) * d.e5_ssrd * KWP_TOT / 1000
        return atm, float(dep[dep.index.month.isin([1, 2, 3, 4, 5, 6]) & (dep.index >= COMMISSION) & dep.index.year.isin(set(H.index.year))].sum())
    return v, mA, mP

def shapley(v, cs=C, part=None):
    cache = {}
    def val(S):
        k = frozenset(S)
        if k not in cache: r = v(tuple(sorted(k))); cache[k] = sum(r) if part is None else r[part]
        return cache[k]
    n = len(cs); phi = {c: 0.0 for c in cs}
    for c in cs:
        oth = [x for x in cs if x != c]
        for r in range(n):
            for S in itertools.combinations(oth, r):
                phi[c] += math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n) * (val(S + (c,)) - val(S))
    return phi, val(tuple(cs))

# ================= N1 trajectory set-ups on a common sample =================
TF = {"p850 (original, 6-hourly, 10-28N 88-110E)": pd.read_csv(f"{TFE_OLD}/tfe_p850.csv", index_col=0, parse_dates=True)}
for k, lab in (("h850", "h850 (hourly)"), ("h850_6h", "h850_6h (hourly data sub-sampled 6-hourly)"), ("blproxy", "blproxy (mean of 10-m and 850-hPa winds)")):
    TF[lab] = pd.read_csv(f"{TFE_NEW}/tfe_{k}.csv", index_col=0, parse_dates=True)
YRS = [2023, 2024, 2025]
common = None
for lab, t in TF.items():
    H, fire = haze_frame(t, YRS); common = H.index if common is None else common.intersection(H.index)
n1 = {}
for lab, t in TF.items():
    H, fire = haze_frame(t, YRS); H = H.loc[H.index.intersection(common)]
    v, mA, mP = engine(H, fire); phi, joint = shapley(v); mA_met = sm.OLS(H.aod_ceres, sm.add_constant(H[MET])).fit()
    mP_met = sm.OLS(np.log(H.pm25), sm.add_constant(H[MET])).fit()
    tt = t[t.index.isin(common)]; ex = tt[[f"tfe_{c}" for c in C]].sum()
    n1[lab] = dict(n=int(len(H)), exposure_share={c: round(100 * ex[f"tfe_{c}"] / ex.sum(), 1) for c in C},
                   frac_valid=round(float(tt.frac_valid.mean()), 3), near24_share_THA=round(100 * tt.tfe24_THA.sum() / tt.tfe_THA.sum(), 1),
                   near24_share_MMR=round(100 * tt.tfe24_MMR.sum() / tt.tfe_MMR.sum(), 1),
                   r2_AOD=round(mA.rsquared, 3), dAIC_AOD=round(mA.aic - mA_met.aic, 1), r2_logPM=round(mP.rsquared, 3), dAIC_logPM=round(mP.aic - mP_met.aic, 1),
                   joint_MWh=round(joint, 1), shapley_share={c: round(100 * phi[c] / joint, 1) for c in C})
    if lab.startswith(("h850 (", "blproxy")):                       # block-bootstrap CI of the atmospheric Shapley share
        hb = (H.index - H.index.min()).days // 14; ub = np.unique(hb); bs = []
        for b in range(200):
            Hr = pd.concat([H[hb == k] for k in rng.choice(ub, len(ub), replace=True)])
            vb, _, _ = engine(Hr, fire, atm_only=True); pb, jb = shapley(vb, part=0); bs.append({c: 100 * pb[c] / jb for c in C})
        BS = pd.DataFrame(bs); n1[lab]["atm_share_CI95"] = {c: [round(BS[c].quantile(0.025), 1), round(BS[c].quantile(0.975), 1)] for c in C}
R["N1_setups_2023_2025"] = n1
old, new = TF["p850 (original, 6-hourly, 10-28N 88-110E)"], TF["h850 (hourly)"]
j = old.join(new, lsuffix="_old", rsuffix="_new", how="inner"); j = j[j.index.isin(common)]
R["N1_daily_corr_log_tfe_old_vs_hourly"] = {c: round(float(np.corrcoef(np.log1p(j[f"tfe_{c}_old"]), np.log1p(j[f"tfe_{c}_new"]))[0, 1]), 2) for c in C}


json.dump(R, open(f"{OUT}/N1_rev25.json", "w"), indent=1, default=float); print(json.dumps(R, indent=1, default=float))
