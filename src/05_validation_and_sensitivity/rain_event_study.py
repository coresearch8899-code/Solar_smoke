"""Rain-cleaning event study: does PV yield recover after rain, and does the recovery scale with the PM2.5 dose accumulated
before the rain (as the dose-reset model assumes)? Residual yield ratio is net of zone, trend, T2m and CERES AOD,
so wet removal of aerosol from the atmosphere is not counted as module cleaning."""
import sys, json, numpy as np, pandas as pd, statsmodels.api as sm
D3, OUT = sys.argv[1:3]; SPELL, WIN, NMIN = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
KWP = {"Z1": 2256.24, "Z2": 207.06, "Z3": 49.90}
CLEAN = pd.to_datetime(["2023-05-06", "2024-03-09", "2024-06-23", "2025-04-25", "2025-12-26", "2026-03-23", "2026-09-05"])
d = pd.read_csv(D3, index_col=0, parse_dates=True)
def dose(thr):
    reset = (d.e5_tp >= thr) | d.index.isin(CLEAN + pd.Timedelta(days=1)); return d.pm25.groupby(reset.cumsum()).cumsum() / 1000.0
d["dose"] = dose(2.0)
rows = []
for z in KWP:
    t = pd.DataFrame({"ratio": d[f"Y_{z}"] / d.e5_ssrd, "aod": d.aod_ceres, "t2m": d.e5_t2m, "soil": d.dose,
                      "yr": (d.index - pd.Timestamp("2023-03-03")).days / 365.25, "zone": z,
                      "clear": (d.e5_tcc_day < 30) & (d.ceres_all / d.ceres_clr > 0.85) & (d.e5_tp < 1),
                      "near": (d.e5_tcc_day < 50) & (d.ceres_all / d.ceres_clr > 0.75) & (d.e5_tp < 1)}, index=d.index)
    med = t.ratio.median(); rows.append(t[t.ratio.between(0.4 * med, 1.3 * med)])
A = pd.concat(rows).dropna(subset=["ratio", "aod", "t2m", "soil"])
def Xf(df):
    X = pd.get_dummies(df.zone).astype(float)
    for z in KWP: X[f"tr_{z}"] = (df.zone == z) * df.yr
    X[["aod", "t2m", "soil"]] = df[["aod", "t2m", "soil"]]; return X
P = A[A.clear]; m0 = sm.OLS(P.ratio, Xf(P)).fit(); B_S, MR = m0.params.soil, P.ratio.mean()
N = A[A.near].copy(); X = Xf(N)
N["res"] = N.ratio - (X.drop(columns="soil") * m0.params.drop("soil")).sum(1)          # soiling term left in the residual
res = N.groupby(N.index).res.mean()                                                    # fleet (zone-averaged) daily residual
soil = d.dose

wet = d.e5_tp >= 2.0; dry1 = d.e5_tp < 1.0
spell = dry1.astype(int).groupby((~dry1).cumsum()).cumsum()                            # length of the current dry spell
def events(mask, label):
    out = []
    for r in d.index[mask]:
        if spell.shift(1).get(r, 0) < SPELL: continue                                     # first rain after >= 5 dry days
        b = res[(res.index >= r - pd.Timedelta(days=WIN)) & (res.index < r)]
        a = res[(res.index > r) & (res.index <= r + pd.Timedelta(days=WIN))]
        if len(b) < NMIN or len(a) < NMIN: continue
        sb = soil[(soil.index >= r - pd.Timedelta(days=WIN)) & (soil.index < r)].mean(); sa = soil.reindex(a.index).mean()
        out.append(dict(date=r, kind=label, rain_mm=round(d.e5_tp[r], 1), dry_days=int(spell.shift(1)[r]), dose_before=sb,
                        obs=100 * (a.mean() - b.mean()) / MR, pred=100 * -B_S * (sb - sa) / MR, n_b=len(b), n_a=len(a),
                        dAOD=d.aod_ceres.reindex(a.index).mean() - d.aod_ceres.reindex(b.index).mean()))
    return pd.DataFrame(out)
ev = events(wet, "rain >= 2 mm")
pl = events(dry1 & (spell >= 6) & (np.arange(len(d)) % 7 == 0), "placebo dry day")      # dry days inside long dry spells
ev.round(3).to_csv(f"{OUT}/N5_rain_events_s{SPELL}w{WIN}n{NMIN}.csv", index=False); pl.round(3).to_csv(f"{OUT}/N5_placebo_s{SPELL}w{WIN}n{NMIN}.csv", index=False)
R = {}
def summ(df):
    if len(df) < 3: return dict(n=len(df))
    fit = sm.OLS(df.obs, sm.add_constant(df.pred)).fit(cov_type="HC1")
    return dict(n=len(df), mean_obs=round(df.obs.mean(), 2), se_obs=round(df.obs.std() / np.sqrt(len(df)), 2), mean_pred=round(df.pred.mean(), 2),
                slope_obs_on_pred=round(fit.params.pred, 2), slope_se=round(fit.bse.pred, 2), slope_p=round(fit.pvalues.pred, 3),
                mean_dAOD=round(df.dAOD.mean(), 3))
R["all_rain_events"] = summ(ev); R["placebo_dry_days"] = summ(pl)
ev["dose_class"] = pd.qcut(ev.dose_before, 3, labels=["low", "mid", "high"])
R["by_pre_rain_dose_tercile"] = {str(k): summ(g) | dict(dose_range=[round(g.dose_before.min(), 2), round(g.dose_before.max(), 2)]) for k, g in ev.groupby("dose_class", observed=True)}
ev["rain_class"] = pd.cut(ev.rain_mm, [2, 5, 10, 1000], right=False, labels=["2-5 mm", "5-10 mm", ">=10 mm"])
R["by_rain_amount"] = {str(k): summ(g) for k, g in ev.groupby("rain_class", observed=True)}
# light rain (1-2 mm) after dry spells: does it already clean? (model assumes no reset below 2 mm)
lt = events((d.e5_tp >= 1.0) & (d.e5_tp < 2.0), "rain 1-2 mm"); R["light_rain_1_2mm"] = summ(lt)
json.dump(R, open(f"{OUT}/N5_rain_event_results_s{SPELL}w{WIN}n{NMIN}.json", "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))

