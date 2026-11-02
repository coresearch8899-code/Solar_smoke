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
def soil_dose(pm, tp, thr=2.0):
    reset = (tp >= thr) | tp.index.isin(CLEAN + pd.Timedelta(days=1)); return pm.groupby(reset.cumsum()).cumsum() / 1000.0
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
        atm = float((-B_AOD * dA * e.e5_ssrd * CLRW.reindex(H.index).values * KWP_TOT / 1000).sum())
        if atm_only: return atm, 0.0
        dP = H.pm25 * (1 - np.exp(-(bP - mP.predict(Zc)).clip(lower=0)))
        pm_c = d.pm25.copy(); pm_c.loc[H.index] = pm_c.loc[H.index] - dP
        dep = -B_SOIL * (d.dose - soil_dose(pm_c, d.e5_tp)) * d.e5_ssrd * KWP_TOT / 1000
        return atm, float(dep[dep.index.month.isin([1, 2, 3, 4, 5, 6]) & dep.index.year.isin(set(H.index.year))].sum())
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

# ================= N2 five-season exposure climatology =================
t = TF["h850 (hourly)"]; t = t[t.index.month.isin([1, 2, 3, 4, 5])]
cl = t.groupby(t.index.year)[[f"tfe_{c}" for c in ["THA", "MMR", "LAO"]]].sum(); cl = (100 * cl.div(cl.sum(1), axis=0)).round(1)
cl.columns = ["THA", "MMR", "LAO"]; cl.to_csv(f"{OUT}/N2_exposure_share_3countries_2021_2025.csv")
R["N2_exposure_share_THA_MMR_LAO"] = cl.to_dict(orient="index")
# five-season PM2.5 haze model is not possible: the daily PM2.5/AOD record (CAMS forecast, CERES) starts in March 2023

# ================= N3 CAMS EAC4 speciation =================
def site_series(f):
    with h5py.File(f, "r") as h:
        lat, lon, tm = h["latitude"][:], h["longitude"][:], pd.to_datetime(h["time"][:], unit="s")
        y = (lat[0] - LAT0) / (lat[0] - lat[1]); x = (LON0 - lon[0]) / (lon[1] - lon[0]); iy, ix = int(y), int(x); fy, fx = y - iy, x - ix
        out = {}
        for v in ("aod550", "omaod550", "bcaod550", "duaod550", "suaod550", "ssaod550", "pm2p5"):
            a = h[v][:, iy:iy + 2, ix:ix + 2].astype(float)
            out[v] = a[:, 0, 0] * (1 - fy) * (1 - fx) + a[:, 1, 0] * fy * (1 - fx) + a[:, 0, 1] * (1 - fy) * fx + a[:, 1, 1] * fy * fx
    s = pd.DataFrame(out, index=tm); s["pm2p5"] *= 1e9; s["smoke"] = s.omaod550 + s.bcaod550
    return s.groupby((s.index + pd.Timedelta(hours=7)).normalize()).mean()        # local-day means
E4 = pd.concat([site_series(f) for f in sorted(glob.glob(f"{EAC4}/cams_eac4_*.nc"))])
E4 = E4[E4.index.year.isin([y_ for y_, n_ in E4.groupby(E4.index.year).size().items() if n_ > 300])]   # local-day shift spills a few hours into the next year
E4.round(4).to_csv(f"{OUT}/N3_eac4_site_daily.csv")
n3 = {}
for y_, g in E4.groupby(E4.index.year):
    bg = g[g.index.month.isin([11, 12, 1])]; hz = g[g.index.month.isin([1, 2, 3, 4, 5])]
    ex_tot = (hz.aod550 - bg.aod550.median()).clip(lower=0); ex_sm = np.minimum((hz.smoke - bg.smoke.median()).clip(lower=0), ex_tot)
    n3[int(y_)] = dict(JanMay_AOD=round(hz.aod550.mean(), 3), JanMay_smoke_AOD=round(hz.smoke.mean(), 3),
                       smoke_frac_of_AOD_JanMay=round(100 * hz.smoke.sum() / hz.aod550.sum(), 1),
                       smoke_frac_of_AOD_excess_JanMay=round(100 * ex_sm.sum() / ex_tot.sum(), 1),
                       dust_frac_JanMay=round(100 * hz.duaod550.sum() / hz.aod550.sum(), 1), sulphate_frac_JanMay=round(100 * hz.suaod550.sum() / hz.aod550.sum(), 1),
                       MarApr_PM25=round(g[g.index.month.isin([3, 4])].pm2p5.mean(), 1))
j4 = E4.join(d[["aod_ceres", "pm25"]], how="inner").dropna()
if len(j4):
    E = pd.DataFrame(index=d.index)
    E["atm"] = -B_AOD * (d.aod_ceres - AOD_BG).clip(lower=0) * d.e5_ssrd * CLRW * KWP_TOT
    k4 = j4[j4.index.month.isin([1, 2, 3, 4, 5])]
    bg4 = E4[(E4.index.year == 2024) & E4.index.month.isin([11, 12, 1])]
    ex_t = (k4.aod550 - bg4.aod550.median()).clip(lower=0); ex_s = np.minimum((k4.smoke - bg4.smoke.median()).clip(lower=0), ex_t)
    fr = (ex_s / ex_t).where(ex_t > 0, 0); w = E.atm.reindex(k4.index).fillna(0)
    f6 = pd.read_csv(f"{S5}/R3_fig6a_components.csv", index_col=0)
    n3["overlap_2024"] = dict(n_days=int(len(j4)), r_AOD_EAC4_vs_CERES=round(float(np.corrcoef(j4.aod550, j4.aod_ceres)[0, 1]), 2),
                              mean_AOD_EAC4=round(j4.aod550.mean(), 3), mean_AOD_CERES=round(j4.aod_ceres.mean(), 3),
                              r_PM25_EAC4_vs_CAMSforecast=round(float(np.corrcoef(j4.pm2p5, j4.pm25)[0, 1]), 2),
                              mean_PM25_EAC4=round(j4.pm2p5.mean(), 1), mean_PM25_CAMSforecast=round(j4.pm25.mean(), 1),
                              MarApr_PM25_EAC4=round(j4[j4.index.month.isin([3, 4])].pm2p5.mean(), 1), MarApr_PM25_CAMSforecast=round(j4[j4.index.month.isin([3, 4])].pm25.mean(), 1),
                              loss_weighted_smoke_share_JanMay=round(100 * float((w * fr).sum() / w.sum()), 1),
                              model_fire_share_JanMay_atm=round(100 * f6.loc[2024, "fire_atm_MWh"] / f6.loc[2024, "atm_JanMay"], 1))
R["N3_eac4"] = n3

# ================= N4 MAIAC vs CERES =================
rows = []
for f in sorted(glob.glob(f"{MAIAC}/maiac_aod055_*_TH.tif")):
    ym = pd.Period(f.split("_")[-2], "M"); a = tifffile.imread(f).astype(float)          # [lat, lon, band] 0.01 deg from 21N, 97E
    r0, c0 = int((21 - LAT0) / 0.01), int((LON0 - 97) / 0.01)
    site = np.nanmean(a[r0 - 5:r0 + 5, c0 - 5:c0 + 5, 0]); cell = np.nanmean(a[100:200, 200:300, 0])     # 0.1 deg window; CERES 1-deg cell 19-20N 99-100E
    nd = np.nanmean(a[r0 - 5:r0 + 5, c0 - 5:c0 + 5, 1])
    ce = d.aod_ceres[(d.index.to_period("M") == ym)]
    rows.append(dict(month=str(ym), maiac_site_0p1=round(site, 3), maiac_1deg_cell=round(cell, 3), maiac_valid_days=round(nd, 1),
                     ceres_month_mean=round(ce.mean(), 3) if len(ce) else np.nan))
M4 = pd.DataFrame(rows).set_index("month"); M4.to_csv(f"{OUT}/N4_maiac_vs_ceres_monthly.csv")
mm = M4.dropna()
R["N4_maiac"] = dict(months_overlap=int(len(mm)), r_site_vs_ceres=round(float(np.corrcoef(mm.maiac_site_0p1, mm.ceres_month_mean)[0, 1]), 2) if len(mm) > 2 else None,
                     mean_ratio_site_to_ceres=round(float((mm.maiac_site_0p1 / mm.ceres_month_mean).mean()), 2) if len(mm) else None,
                     mean_ratio_site_to_1deg_all=round(float((M4.maiac_site_0p1 / M4.maiac_1deg_cell).mean()), 2),
                     ratio_maiac_to_ceres_MarApr=round(float((mm.maiac_site_0p1 / mm.ceres_month_mean)[[m_[-2:] in ("03", "04") for m_ in mm.index]].mean()), 2),
                     ratio_maiac_to_ceres_other=round(float((mm.maiac_site_0p1 / mm.ceres_month_mean)[[m_[-2:] not in ("03", "04") for m_ in mm.index]].mean()), 2),
                     ratio_site_to_1deg_haze_months=round(float((M4.maiac_site_0p1 / M4.maiac_1deg_cell)[[m_[-2:] in ("01", "02", "03", "04", "05") for m_ in M4.index]].mean()), 2))
json.dump(R, open(f"{OUT}/N_newdata_results.json", "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float)); print(M4.to_string()); print(cl.to_string())
