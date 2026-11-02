"""Step 2 – haze/fire–PV analysis with ERA5 (CDS) meteorology, upwind fire exposure and fire land cover.
Python only. Inputs: step-1 daily table, ERA5 site series (era5_extract.py), fires_landcover.csv.gz (landcover.py).
"""
import sys, json, warnings, numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
warnings.filterwarnings("ignore")
S1, E5S, E5P, FLC, OUT = sys.argv[1:6]
LAT, LON, KWP, PR = 19.905, 99.83, 2513.2, 0.80
J2KWH = 1 / 3.6e6
R = {}

# ---------------- A. load ----------------
d1 = pd.read_csv(S1, index_col=0, parse_dates=True)
sl = pd.read_csv(E5S, index_col=0, parse_dates=True)
pl = pd.read_csv(E5P, index_col=0, parse_dates=True)
fl = pd.read_csv(FLC)

# ---------------- B. ERA5 daily ----------------
sl["date"] = sl.index.normalize(); hr = sl.index.hour
e = sl.assign(rh=100 * np.exp(17.625 * (sl.d2m - 273.15) / (243.04 + sl.d2m - 273.15)) /
                 np.exp(17.625 * (sl.t2m - 273.15) / (243.04 + sl.t2m - 273.15)),
              ws=np.hypot(sl.u10, sl.v10), day=(hr >= 10) & (hr <= 16))
agg = {c: (c, lambda s: s.sum() * J2KWH) for c in ("ssrd", "ssrdc", "fdir", "cdir", "tisr")}
E = e.groupby("date").agg(**{f"e5_{k}": v for k, v in agg.items()},
                          e5_tp=("tp", lambda s: s.sum() * 1000), e5_tcwv=("tcwv", "mean"),
                          e5_fal=("fal", "mean"), e5_t2m=("t2m", lambda s: s.mean() - 273.15),
                          e5_rh=("rh", "mean"), e5_ws=("ws", "mean"))
E["e5_blh_day"] = e[e.day].groupby("date").blh.mean()
E["e5_blh_max"] = e.groupby("date").blh.max()
E["e5_tcc_day"] = e[e.day].groupby("date").tcc.mean()
t2m07 = e[hr == 7].set_index("date").t2m
p = pl.copy(); p["date"] = p.index.normalize()
p07 = p[p.index.hour == 7].set_index("date")
E["inv850_07"] = (p07.t850 - t2m07).reindex(E.index)           # K; > -5 => strong morning inversion/stable
E["inv925_07"] = (p07.t925 - t2m07).reindex(E.index)
pw = p.groupby("date")[["u850", "v850", "u700", "v700"]].mean()
for lev in (850, 700):
    E[f"ws{lev}"] = np.hypot(pw[f"u{lev}"], pw[f"v{lev}"])
    E[f"wdir{lev}"] = (np.degrees(np.arctan2(-pw[f"u{lev}"], -pw[f"v{lev}"])) % 360)   # direction wind comes FROM
df = d1.join(E, how="left")
R["era5_check"] = dict(ssrd_mean=float(df.e5_ssrd.mean()), tisr_mean=float(df.e5_tisr.mean()),
                       openmeteo_ghi_mean=float(df.era5_ghi.mean()), tp_annual_2024=float(df.loc["2024", "e5_tp"].sum()),
                       blh_day_mean=float(df.e5_blh_day.mean()))

# ---------------- C. data QC ----------------
qc = {}
for c in ["pm25", "aod_cams", "aod_ceres", "ceres_all", "ceres_clr", "era5_ghi", "e5_ssrd", "e5_ssrdc", "e5_blh_day",
          "inv850_07", "ws850", "frp_RestTH", "oni", "pm25_57T"]:
    s = df[c]; qc[c] = dict(first=str(s.first_valid_index())[:10], last=str(s.last_valid_index())[:10],
                            n=int(s.notna().sum()), pct_missing=round(100 * s.isna().mean(), 1))
pd.DataFrame(qc).T.to_csv(f"{OUT}/T1_data_qc.csv")

# ---------------- D. ERA5 clear-sky bias (aerosol climatology vs CERES actual aerosol) ----------------
c = df[["ceres_clr", "e5_ssrdc", "aod_ceres", "e5_tcwv"]].dropna()
c["ratio"] = c.ceres_clr / c.e5_ssrdc
mb = sm.OLS(c.ratio, sm.add_constant(c[["aod_ceres", "e5_tcwv"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
R["clr_bias"] = dict(b0=mb.params.const, b_aod=mb.params.aod_ceres, se_aod=mb.bse.aod_ceres, b_tcwv=mb.params.e5_tcwv,
                     r2=mb.rsquared, n=int(mb.nobs), ratio_aod_gt1=float(c[c.aod_ceres > 1].ratio.mean()),
                     ratio_aod_lt02=float(c[c.aod_ceres < 0.2].ratio.mean()))
a = df[["ceres_all", "e5_ssrd", "aod_ceres"]].dropna()
R["allsky_bias_haze"] = dict(MAR_APR_ratio=float((a.ceres_all / a.e5_ssrd)[a.index.month.isin([3, 4])].mean()),
                             JUN_SEP_ratio=float((a.ceres_all / a.e5_ssrd)[a.index.month.isin([6, 7, 8, 9])].mean()))

# ---------------- E. transmittance model with water vapour ----------------
df["tau_clr"] = df.ceres_clr / df.etr
t = df[["tau_clr", "aod_ceres", "e5_tcwv"]].dropna()
mt = sm.OLS(t.tau_clr, sm.add_constant(t[["aod_ceres", "e5_tcwv"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
R["tau_model"] = dict(b0=mt.params.const, b_aod=mt.params.aod_ceres, se_aod=mt.bse.aod_ceres,
                      b_tcwv=mt.params.e5_tcwv, se_tcwv=mt.bse.e5_tcwv, r2=mt.rsquared, n=int(mt.nobs))

# ---------------- F. upwind fire exposure ----------------
fl["t"] = pd.to_datetime(fl.acq_date) + pd.to_timedelta(fl.acq_time // 100, "h") + pd.to_timedelta(fl.acq_time % 100, "m") + pd.Timedelta(hours=7)
fl["date"] = fl.t.dt.normalize()
la1, lo1 = np.radians(LAT), np.radians(LON); la2, lo2 = np.radians(fl.latitude), np.radians(fl.longitude)
fl["dist"] = 2 * 6371 * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))
fl["bear"] = np.degrees(np.arctan2(np.sin(lo2 - lo1) * np.cos(la2),
                                   np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(lo2 - lo1))) % 360
nsat = fl.groupby(["date", "sat"]).size().unstack().notna().sum(axis=1)
fl["frp_w"] = fl.frp / fl.date.map(nsat).clip(lower=1)
F = fl[fl.dist < 700]
wd = df.wdir850.dropna()
ufe = {}
for k in (0, 1, 2):
    tmp = F.assign(target=F.date + pd.Timedelta(days=k))
    tmp = tmp[tmp.target.isin(wd.index)]
    ang = np.abs(((tmp.bear - tmp.target.map(wd)) + 180) % 360 - 180)
    w = np.where(ang < 45, np.cos(np.radians(ang * 2)), 0.0)         # cosine kernel within +/-45 deg of upwind
    ufe[k] = (tmp.frp_w * w).groupby(tmp.target).sum().reindex(df.index).fillna(0)
    df[f"ufe_{k}"] = ufe[k]
df["ufe_sum"] = df.ufe_0 + df.ufe_1 + df.ufe_2
# share of upwind exposure that is local (<100 km) vs regional
tmp = F.assign(target=F.date + pd.Timedelta(days=1)); tmp = tmp[tmp.target.isin(wd.index)]
ang = np.abs(((tmp.bear - tmp.target.map(wd)) + 180) % 360 - 180); tmp["w"] = np.where(ang < 45, np.cos(np.radians(ang * 2)), 0) * tmp.frp_w
hz = tmp[tmp.target.dt.month.isin([1, 2, 3, 4, 5])]
_s = hz.groupby(pd.cut(hz.dist, [0, 100, 300, 500, 700])).w.sum() / hz.w.sum(); R["upwind_share_by_distance"] = {str(k): round(float(v), 3) for k, v in _s.items()}

# ---------------- G. land cover of fires ----------------
LC = {10: "Tree", 20: "Shrub", 30: "Grass", 40: "Crop", 50: "Built", 60: "Bare", 80: "Water", 90: "Wetland", 0: "NoData"}
fl["lcn"] = fl.lc.map(LC).fillna("Other")
north = fl[(fl.latitude > 17) & (fl.date.dt.month.isin([1, 2, 3, 4, 5]))]
lc_tab = north.groupby([north.date.dt.year, "lcn"]).frp_w.sum().unstack(fill_value=0)
lc_tab = lc_tab.div(lc_tab.sum(axis=1), axis=0).mul(100).round(1)
lc_tab.to_csv(f"{OUT}/T3_fire_FRP_share_by_landcover_north.csv")
up = tmp[(tmp.target.dt.month.isin([1, 2, 3, 4, 5]))].merge(fl[["lcn"]], left_index=True, right_index=True)
R["upwind_FRP_share_by_landcover_%"] = (up.groupby("lcn").w.sum() / up.w.sum() * 100).round(1).to_dict()

# ---------------- H. PM2.5 models (Jan-May) ----------------
H = df[df.index.month.isin([1, 2, 3, 4, 5])].copy()
for r in ("CR", "UpperN", "RestTH"):
    for L in (0, 1, 2):
        H[f"l{r}_{L}"] = np.log1p(df[f"frp_{r}"].shift(L)).reindex(H.index)
for k in (0, 1, 2): H[f"lufe_{k}"] = np.log1p(H[f"ufe_{k}"])
H["lblh"] = np.log(H.e5_blh_day)
H["s1"] = np.sin(2*np.pi*H.index.dayofyear/365.25); H["c1"] = np.cos(2*np.pi*H.index.dayofyear/365.25)
fire_reg = [f"l{r}_{L}" for r in ("CR", "UpperN", "RestTH") for L in (0, 1, 2)]
met = ["lblh", "inv850_07", "e5_rh", "e5_ws", "e5_tp", "ws850", "e5_tcwv", "s1", "c1"]   # s1/c1: seasonal harmonics
specs = {"M1_fireregion_only": fire_reg, "M2_met_only": met, "M3_fire+met": fire_reg + met,
         "M4_upwind+met": ["lufe_0", "lufe_1", "lufe_2"] + met, "M5_all": fire_reg + ["lufe_0", "lufe_1", "lufe_2"] + met}
mods = {}
for name, x in specs.items():
    Hm = H.dropna(subset=x + ["pm25"])
    m = sm.OLS(np.log(Hm.pm25), sm.add_constant(Hm[x])).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    mods[name] = m
    R.setdefault("pm_models", {})[name] = dict(r2=round(m.rsquared, 3), adj_r2=round(m.rsquared_adj, 3), aic=round(m.aic, 1), n=int(m.nobs))
best = mods["M5_all"]
coef = pd.DataFrame({"coef": best.params, "se": best.bse, "p": best.pvalues}).round(4)
Xs = H.dropna(subset=specs["M5_all"] + ["pm25"])
coef["std_coef"] = (best.params[1:] * Xs[specs["M5_all"]].std()).round(4)
coef.to_csv(f"{OUT}/T4_pm25_model_M5.csv")
# gradient boosting, leave-one-year-out
X = Xs[specs["M5_all"]]; y = np.log(Xs.pm25); yrs = Xs.index.year
pred = pd.Series(index=X.index, dtype=float)
for yr in sorted(set(yrs)):
    gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10, random_state=1)
    gb.fit(X[yrs != yr], y[yrs != yr]); pred[yrs == yr] = gb.predict(X[yrs == yr])
R["gbm_LOYO_r2"] = float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
ols_pred = pd.Series(index=X.index, dtype=float)
for yr in sorted(set(yrs)):
    m = sm.OLS(y[yrs != yr], sm.add_constant(X[yrs != yr])).fit(); ols_pred[yrs == yr] = m.predict(sm.add_constant(X[yrs == yr], has_constant="add"))
R["ols_LOYO_r2"] = float(1 - ((y - ols_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10, random_state=1).fit(X, y)
pi = permutation_importance(gb, X, y, n_repeats=20, random_state=1)
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
grp = {"Fires (regional FRP)": [c for c in X if c.startswith("l") and "_" in c and c[1:4] in ("CR_", "Upp", "Res")],
       "Fires (upwind)": ["lufe_0", "lufe_1", "lufe_2"], "Boundary layer/inversion": ["lblh", "inv850_07"],
       "Moisture/rain": ["e5_rh", "e5_tp", "e5_tcwv"], "Wind": ["e5_ws", "ws850"], "Season": ["s1", "c1"]}
R["gbm_perm_importance_grouped"] = {g: round(float(imp[[c for c in cs if c in imp]].sum()), 3) for g, cs in grp.items()}
imp.round(4).to_csv(f"{OUT}/T5_gbm_permutation_importance.csv")

# ---------------- I. fire-attributable AOD and PV loss ----------------
A = H.dropna(subset=specs["M5_all"] + ["aod_ceres"]).copy()
xa = specs["M5_all"]
ma = sm.OLS(A.aod_ceres, sm.add_constant(A[xa])).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
R["aod_model"] = dict(r2=round(ma.rsquared, 3), n=int(ma.nobs))
bA, bW = mt.params.aod_ceres, mt.params.e5_tcwv
AOD_BG = df.loc[df.index.month.isin([11, 12, 1]), "aod_ceres"].median()
R["aod_bg"] = float(AOD_BG)
def loss_kwh(aod_delta, rows):          # extra PV energy lost because of aod_delta (>0), all-sky scaling
    dH = -bA * aod_delta * rows.etr * (rows.ceres_all / rows.ceres_clr).clip(0, 1)
    return dH * KWP * PR
base = ma.predict(sm.add_constant(A[xa]))
fire_cols = [c for c in xa if c.startswith("lCR") or c.startswith("lUpperN") or c.startswith("lRestTH") or c.startswith("lufe")]
att = {}
for label, cols in {"All fires": fire_cols, "Chiang Rai": [c for c in fire_cols if c.startswith("lCR")],
                    "Upper North (ex-CR)": [c for c in fire_cols if c.startswith("lUpperN")],
                    "Rest of Thailand": [c for c in fire_cols if c.startswith("lRestTH")],
                    "Upwind index": [c for c in fire_cols if c.startswith("lufe")]}.items():
    Z = A[xa].copy()
    for cc in cols: Z[cc] = Z[cc].clip(upper=A[cc].quantile(0.10))     # counterfactual: low-fire (P10) days
    d_aod = (base - ma.predict(sm.add_constant(Z, has_constant="add"))).clip(lower=0)
    d_aod = np.minimum(d_aod, (A.aod_ceres - AOD_BG).clip(lower=0))      # cannot remove more than the excess over clean background
    A[f"dAOD_{label}"] = d_aod
    A[f"dE_{label}"] = loss_kwh(d_aod, A)
yr = A.groupby(A.index.year)
att_tab = pd.DataFrame({lab: yr[f"dE_{lab}"].sum() / 1000 for lab in ["All fires", "Chiang Rai", "Upper North (ex-CR)", "Rest of Thailand", "Upwind index"]}).round(1)
A["dE_total_excess"] = loss_kwh((A.aod_ceres - AOD_BG).clip(lower=0), A)
att_tab["Total aerosol excess"] = (yr["dE_total_excess"].sum() / 1000).round(1)
att_tab["fire_share_%"] = (100 * att_tab["All fires"] / att_tab["Total aerosol excess"]).round(1)
att_tab["mean_dAOD_all"] = yr["dAOD_All fires"].mean().round(3)
att_tab["n_days"] = yr.size()
att_tab.to_csv(f"{OUT}/T6_fire_attributable_PV_loss_MWh.csv")

# ---------------- J. season x ENSO ----------------
ma_ = df[df.index.month.isin([3, 4])]
season = ma_.groupby(ma_.index.year).agg(pm25=("pm25", "mean"), aod=("aod_ceres", "mean"), blh=("e5_blh_day", "mean"),
                                         inv850=("inv850_07", "mean"), rain_mm=("e5_tp", "sum"), ws850=("ws850", "mean"),
                                         frp_north=("frp_UpperN", "sum"), ufe=("ufe_1", "sum"), oni=("oni", "mean")).round(2)
season.to_csv(f"{OUT}/T7_MarApr_season_ENSO.csv")
binned = H.groupby(pd.qcut(H.e5_blh_day, 5)).pm25.mean().round(1)
R["pm_by_blh_quintile"] = {str(k): float(v) for k, v in binned.items()}
binned2 = H.groupby(pd.qcut(H.inv850_07, 5)).pm25.mean().round(1)
R["pm_by_inv850_quintile"] = {str(k): float(v) for k, v in binned2.items()}

df.round(5).to_csv(f"{OUT}/CRH_haze_daily_step2.csv")
print(json.dumps(R, indent=1, default=float))
print("\nT4 M5 coefficients\n", coef.to_string())
print("\nT6 fire-attributable PV loss (MWh, Jan-May)\n", att_tab.to_string())
print("\nT7 Mar-Apr season\n", season.to_string())
print("\nT3 FRP share by land cover (north >17N, Jan-May)\n", lc_tab.to_string())
print("\nT1 QC\n", pd.DataFrame(qc).T.to_string())
