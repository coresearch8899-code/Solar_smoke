"""rev25 extras with the partial-cleaning dose: Table 3 (HAC standard errors, zone trends, R2), GBM check of the AOD response,
dose statistics (Jan-May mean, largest modelled loss, soiling rate per day), and Table S10 (hourly-wind set-ups; the N1 block of
haze_pv_step6_newdata.py re-run with the rev25 dose).
Usage: extras_rev25.py <step3_csv> <step3_dir_with_tfe> <step6_dir_with_hourly_tfe> <step6_script> <out_dir>"""
import sys, json, re, warnings, numpy as np, pandas as pd, statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")
D3, TFE_OLD, TFE_NEW, STEP6, OUT = sys.argv[1:6]
KWP = {"Z1": 2256.24, "Z2": 207.06, "Z3": 49.90}; KWP_TOT = sum(KWP.values())
CLEAN = pd.to_datetime(["2023-05-06", "2024-03-09", "2024-06-23", "2025-04-25", "2025-12-26", "2026-03-23", "2026-09-05"])
COMMISSION = pd.Timestamp("2023-03-03"); PF = 2.0
d = pd.read_csv(D3, index_col=0, parse_dates=True)
def soil_dose(pm, tp, thr=PF, clean=CLEAN):
    wash = pm.index.isin(clean + pd.Timedelta(days=1)) | (pm.index == COMMISSION)
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
                      "yr": (d.index - COMMISSION).days / 365.25, "zone": z}, index=d.index)
    med = t.ratio.median(); rows.append(t[t.clear & t.ratio.between(0.4 * med, 1.3 * med)])
PV = pd.concat(rows).drop(columns="clear").dropna()
X = pd.get_dummies(PV.zone).astype(float)
for z in KWP: X[f"tr_{z}"] = (PV.zone == z) * PV.yr
X[["aod", "t2m", "soil"]] = PV[["aod", "t2m", "soil"]]
m = sm.OLS(PV.ratio, X).fit(cov_type="HAC", cov_kwds={"maxlags": 7}); MR = PV.ratio.mean()
R = dict(table3=dict(n=int(m.nobs), r2=m.rsquared, aic=m.aic, mean_ratio=MR,
                     coef={k: m.params[k] for k in m.params.index}, se={k: m.bse[k] for k in m.bse.index}, p={k: m.pvalues[k] for k in m.pvalues.index}))
gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=1).fit(X, PV.ratio)
Xa, Xb = X.copy(), X.copy(); Xa["aod"] = 0.2; Xb["aod"] = 1.0
R["gbm_delta_ratio_aod_0.2_to_1.0"] = float((gb.predict(Xb) - gb.predict(Xa)).mean()); R["ols_delta_ratio_aod_0.2_to_1.0"] = float(0.8 * m.params.aod)
# dose statistics
jm = d.index.month.isin([1, 2, 3, 4, 5]) & (d.index >= COMMISSION)
R["dose_JanMay_mean"] = float(d.dose[jm].mean())
loss_pct = -100 * m.params.soil * d.dose / MR
R["max_modelled_loss_pct"] = float(loss_pct[d.index >= COMMISSION].max()); R["max_modelled_loss_date"] = str(loss_pct[d.index >= COMMISSION].idxmax().date())
R["loss_pct_JanJun2024_max"] = float(loss_pct["2024-01-01":"2024-06-30"].max())
for lab, mm in (("JanMay", [1, 2, 3, 4, 5]), ("JunDec", [6, 7, 8, 9, 10, 11, 12])):
    sel = d.index.month.isin(mm) & (d.index >= COMMISSION)
    R[f"soiling_rate_pct_per_day_{lab}"] = float(-100 * m.params.soil * d.pm25[sel].mean() / 1000 / MR)
    R[f"pm25_mean_{lab}"] = float(d.pm25[sel].mean())
R["days_rain_ge2mm_per_year"] = float((d.e5_tp[d.index >= COMMISSION] >= 2).groupby(d.index[d.index >= COMMISSION].year).sum().loc[[2024, 2025]].mean())
R["days_rain_0p1_to_2mm_per_year"] = float(((d.e5_tp >= 0.1) & (d.e5_tp < 2))[d.index >= COMMISSION].groupby(d.index[d.index >= COMMISSION].year).sum().loc[[2024, 2025]].mean())
pd.DataFrame({"date": d.index, "dose": d.dose, "loss_pct": loss_pct, "tp": d.e5_tp}).to_csv(f"{OUT}/dose_series_rev25.csv", index=False)
json.dump(R, open(f"{OUT}/extras_rev25.json", "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))

# ---------- Table S10: N1 block of step 6 with the rev25 dose ----------
src = open(STEP6, encoding="utf-8").read()
src = src[: src.index("# ================= N2")]
old = re.search(r"def soil_dose\(pm, tp, thr=2\.0\):\n.*?\n(?=d\[\"dose\"\])", src, flags=re.S).group(0)
new = ('COMMISSION = pd.Timestamp("2023-03-03"); PF = 2.0\n'
       'def soil_dose(pm, tp, thr=PF):\n'
       '    wash = pm.index.isin(CLEAN + pd.Timedelta(days=1)) | (pm.index == COMMISSION)\n'
       '    keep = 1 - np.minimum(1.0, tp.reindex(pm.index).fillna(0).values / thr); x = pm.values / 1000.0\n'
       '    out = np.full(len(pm), np.nan); D = 0.0\n'
       '    for i in range(len(pm)):\n'
       '        D = 0.0 if wash[i] else D * keep[i]\n'
       '        if not np.isnan(x[i]): D += x[i]; out[i] = D\n'
       '    return pd.Series(out, index=pm.index)\n')
src = src.replace(old, new).replace("KWP_TOT / 1000).sum())", "KWP_TOT / 1000 * (H.index >= COMMISSION)).sum())").replace("dep.index.month.isin([1, 2, 3, 4, 5, 6]) & dep.index.year.isin", "dep.index.month.isin([1, 2, 3, 4, 5, 6]) & (dep.index >= COMMISSION) & dep.index.year.isin") + '\njson.dump(R, open(f"{OUT}/N1_rev25.json", "w"), indent=1, default=float); print(json.dumps(R, indent=1, default=float))\n'
open(f"{OUT}/step6_N1_rev25.py", "w", encoding="utf-8").write(src)
sys.argv = ["step6_N1_rev25.py", D3, TFE_OLD, TFE_NEW, "-", "-", "-", OUT]
exec(compile(src, "step6_N1_rev25.py", "exec"), {"__name__": "__main__"})
