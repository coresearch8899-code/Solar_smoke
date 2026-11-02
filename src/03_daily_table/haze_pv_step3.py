"""Step 3 – ground-truth PV decomposition (atmospheric vs deposition), on-site sensor check,
trajectory-based country attribution of haze and PV loss. Python only."""
import sys, json, warnings, numpy as np, pandas as pd, statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")
S2, PVD, MET, TFE_DIR, OUT = sys.argv[1:6]
import os; CF_ENV = os.environ.get("CF_MODE", "p10")
KWP = {"Z1": 2256.24, "Z2": 207.06, "Z3": 49.90}; KWP_TOT = sum(KWP.values())
CLEAN = pd.to_datetime(["2023-05-06", "2024-03-09", "2024-06-23", "2025-04-25", "2025-12-26", "2026-03-23", "2026-09-05"])
R = {}
hac = dict(cov_type="HAC", cov_kwds={"maxlags": 7})

# ---------------- load ----------------
d = pd.read_csv(S2, index_col=0, parse_dates=True)
pv = pd.read_csv(PVD, encoding="utf-8-sig", parse_dates=["date"])
pv["z"] = pv.zone.str.extract(r"(Zone\d)")[0].str.replace("Zone", "Z")
Y = pv.pivot_table(index="date", columns="z", values="daily_production_kwh")
for z_ in KWP: Y[z_] = Y[z_] / KWP[z_]                                      # specific yield kWh/kWp
met = pd.read_csv(MET, index_col=0, parse_dates=True)
tfe = {k: pd.read_csv(f"{TFE_DIR}/tfe_{k}.csv", index_col=0, parse_dates=True) for k in ("p850", "p700", "3d850")}
C = ["THA", "MMR", "LAO", "OTH"]

# ---------------- A. QC ----------------
qc = pd.DataFrame({
    "PV daily (3 zones)": [str(pv.date.min().date()), str(pv.date.max().date()), int(pv.date.nunique()),
                           round(100 * (pv.daily_production_kwh.fillna(0) <= 0).mean(), 1)],
    "On-site meteo (Zone1, 15 min)": [str(met.index.min().date()), str(met.index.max().date()), int((met.n_valid_ghi >= 80).sum()),
                                     round(100 * (met.n_valid_ghi < 80).mean(), 1)],
    "Trajectories p850 (Jan-May)": [str(tfe["p850"].index.min().date()), str(tfe["p850"].index.max().date()), len(tfe["p850"]),
                                    round(100 * (1 - tfe["p850"].frac_valid.mean()), 1)]},
    index=["first", "last", "n_days_ok", "pct_flagged"]).T
qc.to_csv(f"{OUT}/T8_new_data_qc.csv")

# ---------------- B. on-site pyranometer vs satellite ----------------
m = met.join(d[["ceres_all", "e5_ssrd", "aod_ceres", "e5_tcc_day"]], how="inner")
m = m[(m.n_valid_ghi >= 80) & (m.ghi_int_kWh > 0.5)]
m["r_ceres"] = m.ghi_int_kWh / m.ceres_all; m["r_era5"] = m.ghi_int_kWh / m.e5_ssrd
m["yr"] = (m.index - m.index[0]).days / 365.25
clr = m[(m.e5_tcc_day < 30) & (m.aod_ceres < 0.4)]
dr = sm.OLS(clr.r_era5, sm.add_constant(clr.yr)).fit(**hac)
dr2 = sm.OLS(m.r_ceres, sm.add_constant(m.yr)).fit(**hac)
R["sensor_drift"] = dict(clean_clear_days=int(len(clr)), ratio_era5_start=dr.params.const, drift_per_yr=dr.params.yr,
                         drift_pct_per_yr=100 * dr.params.yr / dr.params.const, se=dr.bse.yr,
                         ceres_ratio_drift_pct_per_yr=100 * dr2.params.yr / dr2.params.const)

# ---------------- C. haze models with trajectory fire exposure ----------------
T0 = tfe["p850"]
H = d.join(T0[[f"tfe_{c}" for c in C] + ["frac_valid"]], how="inner")
H = H[H.index.month.isin([1, 2, 3, 4, 5])].copy()
H["lblh"] = np.log(H.e5_blh_day); doy = H.index.dayofyear
H["s1"], H["c1"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
for c in C:
    for L in (0, 1):
        H[f"lt_{c}_{L}"] = np.log1p(H[f"tfe_{c}"].shift(L))
met_x = ["lblh", "inv850_07", "e5_rh", "e5_ws", "e5_tp", "ws850", "e5_tcwv", "s1", "c1"]
fire_x = [f"lt_{c}_{L}" for c in C for L in (0, 1)]
Hm = H.dropna(subset=fire_x + met_x + ["pm25", "aod_ceres"])
mods = {}
for y_, lab in ((np.log(Hm.pm25), "logPM25"), (Hm.aod_ceres, "AOD")):
    for nm, xs in (("met", met_x), ("traj+met", fire_x + met_x)):
        mm = sm.OLS(y_, sm.add_constant(Hm[xs])).fit(**hac); mods[(lab, nm)] = mm
        R.setdefault("haze_models", {})[f"{lab}|{nm}"] = dict(r2=round(mm.rsquared, 3), adj=round(mm.rsquared_adj, 3), aic=round(mm.aic, 1), n=int(mm.nobs))
    # leave-one-year-out
    yrs = Hm.index.year; pr = pd.Series(index=Hm.index, dtype=float)
    for yy in sorted(set(yrs)):
        mm = sm.OLS(y_[yrs != yy], sm.add_constant(Hm.loc[yrs != yy, fire_x + met_x])).fit()
        pr[yrs == yy] = mm.predict(sm.add_constant(Hm.loc[yrs == yy, fire_x + met_x], has_constant="add"))
    R["haze_models"][f"{lab}|LOYO_r2"] = round(float(1 - ((y_ - pr) ** 2).sum() / ((y_ - y_.mean()) ** 2).sum()), 3)
coefPM = pd.DataFrame({"coef": mods[("logPM25", "traj+met")].params, "se": mods[("logPM25", "traj+met")].bse,
                       "p": mods[("logPM25", "traj+met")].pvalues}).round(4)
coefAOD = pd.DataFrame({"coef": mods[("AOD", "traj+met")].params, "se": mods[("AOD", "traj+met")].bse,
                        "p": mods[("AOD", "traj+met")].pvalues}).round(4)
pd.concat({"logPM25": coefPM, "AOD": coefAOD}, axis=1).to_csv(f"{OUT}/T9_haze_models_trajectory.csv")

CF_MODE = CF_ENV
def counterfactual(model, X, cols):
    Z_ = X.copy()
    for cc in cols: Z_[cc] = np.minimum(Z_[cc], X[cc].quantile(0.10)) if CF_MODE == "p10" else 0.0
    return model.predict(sm.add_constant(Z_, has_constant="add"))

# ---------------- D. PV ground-truth model ----------------
P = d[["e5_ssrd", "e5_tcc_day", "ceres_all", "ceres_clr", "aod_ceres", "pm25", "e5_tp", "e5_t2m"]].join(Y, how="inner")
P = P.join(met[["t_amb"]], how="left")
P["clrfrac"] = P.ceres_all / P.ceres_clr
def soiling(thr):
    reset_rain = P.e5_tp >= thr
    reset = reset_rain | P.index.isin(CLEAN + pd.Timedelta(days=1))
    grp = reset.cumsum()
    dose = P.pm25.groupby(grp).cumsum() / 1000.0                          # mg m-3 d
    days = P.groupby(grp).cumcount().astype(float)
    return dose, days
long = []
for z_ in KWP:
    t_ = P[[z_, "e5_ssrd", "e5_tcc_day", "clrfrac", "aod_ceres", "e5_t2m", "pm25", "e5_tp"]].rename(columns={z_: "y"}).copy()
    t_["zone"] = z_; long.append(t_)
L = pd.concat(long)
L["ratio"] = L.y / L.e5_ssrd
L["yr"] = (L.index - pd.Timestamp("2023-03-03")).days / 365.25
L["clear"] = (L.e5_tcc_day < 30) & (L.clrfrac > 0.85) & (L.e5_tp < 1)
med = L.groupby("zone").ratio.transform("median")
L["ok"] = L.ratio.between(0.4 * med, 1.3 * med) & (L.y > 0)               # drop outages/curtailment/clipping artefacts
best = None; scan = {}
for thr in (2, 5, 10, 20):
    dose, days = soiling(thr)
    L["dose"] = L.index.map(dose); L["days"] = L.index.map(days)
    D_ = L[L.clear & L.ok].dropna(subset=["aod_ceres", "dose", "e5_t2m"]).copy()
    X0 = pd.get_dummies(D_.zone, drop_first=False).astype(float)
    for z_ in KWP: X0[f"trend_{z_}"] = (D_.zone == z_) * D_.yr
    X0["aod"] = D_.aod_ceres; X0["t2m"] = D_.e5_t2m
    for kind in ("dose", "days"):
        X = X0.assign(soil=D_[kind])
        mm = sm.OLS(D_.ratio, X).fit(**hac)
        scan[f"{kind}|rain>={thr}mm"] = dict(aic=round(mm.aic, 1), r2=round(mm.rsquared, 3), b_soil=mm.params.soil, p_soil=mm.pvalues.soil,
                                            b_aod=mm.params.aod, se_aod=mm.bse.aod, n=int(mm.nobs))
        if best is None or mm.aic < best[0]: best = (mm.aic, thr, kind, mm, D_, X)
R["pv_model_scan"] = scan
_, THR, KIND, MPV, DPV, XPV = best
R["pv_best"] = dict(rain_reset_mm=THR, soiling_metric=KIND, r2=MPV.rsquared, n=int(MPV.nobs), b_aod=MPV.params.aod, se_aod=MPV.bse.aod,
                    b_soil=MPV.params.soil, se_soil=MPV.bse.soil, b_t2m=MPV.params.t2m, mean_ratio=float(DPV.ratio.mean()))
pd.DataFrame({"coef": MPV.params, "se": MPV.bse, "p": MPV.pvalues}).round(5).to_csv(f"{OUT}/T10_pv_clear_day_model.csv")
# relative effects: % of yield per unit AOD and per soiling unit
R["pv_best"]["pct_per_unit_aod"] = 100 * MPV.params.aod / DPV.ratio.mean()
R["pv_best"]["pct_per_soil_unit"] = 100 * MPV.params.soil / DPV.ratio.mean()
# GBM robustness check of AOD effect (partial dependence slope)
gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=1).fit(XPV, DPV.ratio)
Xa, Xb = XPV.copy(), XPV.copy(); Xa["aod"] = 0.2; Xb["aod"] = 1.0
R["pv_best"]["gbm_delta_ratio_aod_0.2_to_1.0"] = float((gb.predict(Xb) - gb.predict(Xa)).mean())
R["pv_best"]["ols_delta_ratio_aod_0.2_to_1.0"] = float(0.8 * MPV.params.aod)

# ---------------- E. fleet loss decomposition (all days, all-sky scaling) ----------------
AOD_BG = d.loc[d.index.month.isin([11, 12, 1]), "aod_ceres"].median()
dose, days = soiling(THR); soil = (dose if KIND == "dose" else days)
E = P[["e5_ssrd", "clrfrac", "aod_ceres", "pm25"]].copy(); E["soil"] = soil
E["clrw"] = E.clrfrac.clip(0, 1)                                            # aerosol effect scales with direct-beam share
E["dE_atm_kWh"] = -MPV.params.aod * (E.aod_ceres - AOD_BG).clip(lower=0) * E.e5_ssrd * E.clrw * KWP_TOT
E["dE_dep_kWh"] = -MPV.params.soil * E.soil * E.e5_ssrd * KWP_TOT
E["E_act_kWh"] = P[list(KWP)].mul(pd.Series(KWP)).sum(1, min_count=3)
E["year"] = E.index.year; E["haze"] = E.index.month.isin([1, 2, 3, 4, 5])
dec = E.groupby(["year"]).agg(E_actual_MWh=("E_act_kWh", lambda s: s.sum() / 1000), atm_MWh=("dE_atm_kWh", lambda s: s.sum() / 1000),
                             dep_MWh=("dE_dep_kWh", lambda s: s.sum() / 1000), days=("E_act_kWh", "count"))
dec_h = E[E.haze].groupby("year").agg(atm_MWh_JanMay=("dE_atm_kWh", lambda s: s.sum() / 1000), dep_MWh_JanMay=("dE_dep_kWh", lambda s: s.sum() / 1000))
dec = dec.join(dec_h)
dec["loss_pct"] = 100 * (dec.atm_MWh + dec.dep_MWh) / (dec.E_actual_MWh + dec.atm_MWh + dec.dep_MWh)
dec.round(1).to_csv(f"{OUT}/T11_pv_loss_decomposition.csv")

# ---------------- F. country attribution (Jan-May) ----------------
mA, mP = mods[("AOD", "traj+met")], mods[("logPM25", "traj+met")]
X = Hm[fire_x + met_x]; baseA = mA.predict(sm.add_constant(X)); baseP = mP.predict(sm.add_constant(X))
att = {}
for c in C + ["ALL"]:
    cols = [x for x in fire_x if (c == "ALL" or f"_{c}_" in x)]
    dA = (baseA - counterfactual(mA, X, cols)).clip(lower=0)
    dA = np.minimum(dA, (Hm.aod_ceres - AOD_BG).clip(lower=0))
    pm_cf = Hm.pm25 * np.exp(-(baseP - counterfactual(mP, X, cols)).clip(lower=0))
    att[c] = dict(dAOD=dA, dPM=Hm.pm25 - pm_cf)
rows = {}
for c, v in att.items():
    e = E.reindex(Hm.index)
    atm = -MPV.params.aod * v["dAOD"] * e.e5_ssrd * e.clrw * KWP_TOT
    # deposition: soiling state rebuilt with counterfactual PM (dose metric) – only meaningful if KIND == 'dose'
    if KIND == "dose":
        pm_c = P.pm25.copy(); ix_ = Hm.index.intersection(P.index); pm_c.loc[ix_] = pm_c.loc[ix_] - v["dPM"].loc[ix_]
        reset = (P.e5_tp >= THR) | P.index.isin(CLEAN + pd.Timedelta(days=1)); grp = reset.cumsum()
        soil_c = pm_c.groupby(grp).cumsum() / 1000.0
        dep = -MPV.params.soil * (soil - soil_c) * P.e5_ssrd * KWP_TOT
        dep = dep[dep.index.month.isin([1, 2, 3, 4, 5, 6])]
    else:
        dep = pd.Series(0.0, index=Hm.index)
    rows[c] = pd.DataFrame({"atm_MWh": atm.groupby(atm.index.year).sum() / 1000, "dep_MWh": dep.groupby(dep.index.year).sum() / 1000,
                            "dPM_mean": v["dPM"].groupby(v["dPM"].index.year).mean(), "dAOD_mean": v["dAOD"].groupby(v["dAOD"].index.year).mean()})
attr = pd.concat(rows, axis=1).round(2)
attr.to_csv(f"{OUT}/T12_country_attribution{'' if CF_MODE == 'p10' else '_fullremoval'}.csv")
R["tfe_P10_JanMay"] = {c: float(Hm[f"tfe_{c}"].quantile(0.10)) for c in C}
R["tfe_zero_days_pct"] = {c: float(100 * (Hm[f"tfe_{c}"] == 0).mean()) for c in C}
share = pd.DataFrame({c: T0[f"tfe_{c}"].groupby(T0.index.year).sum() for c in C}); share = 100 * share.div(share.sum(1), axis=0)
share.round(1).to_csv(f"{OUT}/T13_trajectory_fire_exposure_share.csv")
share3 = {k: (lambda s: (100 * s / s.sum()).round(1).to_dict())(v[[f"tfe_{c}" for c in C]].sum()) for k, v in tfe.items()}
R["tfe_share_by_setup"] = share3

# ---------------- G. merged output ----------------
out = d.join(T0[[f"tfe_{c}" for c in C]], how="left").join(Y.add_prefix("Y_"), how="left").join(met.add_prefix("site_"), how="left")
out["soil_index"] = soil; out["dE_atm_kWh"] = E.dE_atm_kWh; out["dE_dep_kWh"] = E.dE_dep_kWh
out.round(5).to_csv(f"{OUT}/CRH_haze_daily_step3.csv")
print(json.dumps(R, indent=1, default=float))
print("\nT9 logPM25 & AOD (trajectory+met)\n", pd.concat({"logPM25": coefPM, "AOD": coefAOD}, axis=1).to_string())
print("\nT10 PV model\n", pd.read_csv(f"{OUT}/T10_pv_clear_day_model.csv", index_col=0).to_string())
print("\nT11 loss decomposition\n", dec.round(1).to_string())
print("\nT12 attribution\n", attr.to_string())
print("\nT13 TFE share\n", share.round(1).to_string())
print("\nT8 QC\n", qc.to_string())
