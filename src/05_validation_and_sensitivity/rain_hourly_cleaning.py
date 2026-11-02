"""Hourly test of rain cleaning (natural cleaning) of the PV fleet.
Inputs: 30-min zone power (monitoring platform), hourly ERA5 weather and CAMS PM2.5/AOD at the site (Open-Meteo, local time),
daily step-3 table (CERES AOD, daily dose). Output: model comparison of rain-reset rules on clear hours, and an event study.
H0 timing check | H1 clear-hour models: no soiling, daily 2-mm reset, hourly event resets P* = 0.5-10 mm, partial cleaning
H2 event study around rain events (72 h before / after) with placebo pseudo-events."""
import sys, json, numpy as np, pandas as pd, statsmodels.api as sm
PVF, WX, AQ, D3, OUT = sys.argv[1:6]
CAMS_CTRL = len(sys.argv) > 6 and sys.argv[6] == "cams"                  # event study also controls hourly CAMS AOD
TAG = "_camsctrl" if CAMS_CTRL else ""
KWP = {"Zone1": 2256.24, "Zone2": 207.06, "Zone3": 49.90}
CLEAN = pd.to_datetime(["2023-05-06", "2024-03-09", "2024-06-23", "2025-04-25", "2025-12-26", "2026-03-23", "2026-09-05"])
R = {}

# ---------------- data ----------------
p = pd.read_csv(PVF, usecols=[1, 2, 22], encoding="utf-8"); p.columns = ["zone", "time", "kw"]
p["zone"] = p.zone.str.extract(r"(Zone\d)")[0]; p["time"] = pd.to_datetime(p.time)
P30 = p.pivot_table(index="time", columns="zone", values="kw")
P30 = P30.reindex(pd.date_range(P30.index.min(), P30.index.max(), freq="30min"))
# hourly mean power over (T-1h, T] from the 30-min samples (trapezoid), matching Open-Meteo 'preceding hour' radiation
Ph = (P30.shift(2) + 2 * P30.shift(1) + P30) / 4; Ph = Ph[Ph.index.minute == 0]
wx = pd.read_csv(WX, skiprows=3, parse_dates=["time"]).set_index("time")
wx.columns = ["t2m", "rh", "prcp", "cc", "ghi", "dni_h", "dif", "ws", "wd"]
aq = pd.read_csv(AQ, skiprows=3, parse_dates=["time"]).set_index("time"); aq.columns = ["pm25", "pm10", "co", "aod_cams", "dust"]
d = pd.read_csv(D3, index_col=0, parse_dates=True)
H = Ph.join(wx, how="inner").join(aq[["pm25", "aod_cams"]], how="left")
H["date"] = H.index.normalize(); H["aod_ceres"] = H.date.map(d.aod_ceres); H["dose_daily"] = H.date.map(d.dose if "dose" in d else d.soil_index)

# ---------------- H0 timing check ----------------
lag = {}
for L in (-1, 0, 1):
    x = pd.concat([Ph["Zone1"], wx.ghi.shift(L)], axis=1).dropna(); x = x[x.ghi > 50]; lag[L] = round(float(x.corr().iloc[0, 1]), 4)
R["H0_corr_power_vs_ghi_by_lag_h"] = lag

# ---------------- rain events and hourly dose ----------------
wet = H.prcp >= 0.1
ev_id = (wet & ~wet.shift(1, fill_value=False)).cumsum().where(wet)
# merge events separated by <= 2 dry hours
gap = (~wet).astype(int).groupby(wet.cumsum()).cumsum()
start = wet & ~(wet.shift(1, fill_value=False) | (wet.shift(2, fill_value=False) & (gap.shift(1) <= 1)) | (wet.shift(3, fill_value=False) & (gap.shift(1) <= 2)))
eid = start.cumsum().where(wet)
EV = H[wet].assign(eid=eid[wet]).groupby("eid").agg(t0=("prcp", lambda s: s.index.min()), t1=("prcp", lambda s: s.index.max()), P=("prcp", "sum"))
EV["dry_before_h"] = [(t - H.index[(H.index < t) & wet].max()).total_seconds() / 3600 if ((H.index < t) & wet).any() else np.nan for t in EV.t0]
R["events"] = dict(n=int(len(EV)), n_ge1mm=int((EV.P >= 1).sum()), n_ge2mm=int((EV.P >= 2).sum()), n_ge5mm=int((EV.P >= 5).sum()))
pm = H.pm25.fillna(H.pm25.median()).values / 24 / 1000                  # mg m-3 d per hour
end_idx = {H.index.get_loc(t1): P for t1, P in zip(EV.t1, EV.P)}
wash_idx = set(np.where(H.index.isin(CLEAN + pd.Timedelta(days=1)))[0])
def hourly_dose(rule, par):
    D = np.zeros(len(H)); cur = 0.0
    for i in range(len(H)):
        if i in wash_idx: cur = 0.0
        cur += pm[i]
        if i in end_idx:                                                  # apply cleaning at the end of a rain event
            P = end_idx[i]
            if rule == "full" and P >= par: cur = 0.0
            elif rule == "partial": cur *= max(0.0, 1 - P / par)
        D[i] = cur
    return pd.Series(D, index=H.index)

# ---------------- H1 clear-hour models ----------------
rows = []
for z, k in KWP.items():
    t = pd.DataFrame({"ratio": H[z] / (k * H.ghi / 1000), "zone": z, "hour": H.index.hour, "month": H.index.month, "t2m": H.t2m,
                      "aod": H.aod_ceres, "aod_cams": H.aod_cams, "dose_daily": H.dose_daily,
                      "yr": (H.index - pd.Timestamp("2023-03-03")).days / 365.25, "date": H.date}, index=H.index)
    rain_near = (H.prcp.rolling(3, min_periods=1).sum() > 0)
    t["clear"] = (H.cc < 20) & (H.ghi > 300) & ~rain_near & ~H.date.isin(CLEAN) & H.index.hour.isin(range(9, 17))
    rows.append(t)
L = pd.concat(rows); L = L[L.clear].dropna(subset=["ratio", "aod", "t2m"])
L["cell"] = L.zone + "_" + L.hour.astype(str) + "_" + L.month.astype(str)
med = L.groupby("cell").ratio.transform("median"); L = L[L.ratio.between(0.6 * med, 1.25 * med)]
FE = pd.get_dummies(L.cell, drop_first=False).astype(float)
for z in KWP: FE[f"tr_{z}"] = (L.zone == z) * L.yr
base = FE.assign(aod=L.aod, t2m=L.t2m)
groups = pd.factorize(L.date)[0]
def fit(soil=None, extra=None):
    X = base.copy()
    if extra is not None: X = X.assign(**extra)
    if soil is not None: X["soil"] = soil.reindex(L.index).values
    return sm.OLS(L.ratio, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
MR = L.ratio.mean()
res = {}; m_none = fit(); res["no soiling term"] = dict(aic=m_none.aic)
m_d = fit(L.dose_daily); res["daily dose, daily rain >= 2 mm (paper)"] = dict(aic=m_d.aic, b=m_d.params.soil, se=m_d.bse.soil, p=m_d.pvalues.soil)
doses = {}
for Pstar in (0.5, 1, 2, 3, 5, 10):
    D = hourly_dose("full", Pstar); doses[("full", Pstar)] = D; m = fit(D)
    res[f"hourly dose, event reset P >= {Pstar} mm"] = dict(aic=m.aic, b=m.params.soil, se=m.bse.soil, p=m.pvalues.soil)
for Pf in (2, 5, 10, 20):
    D = hourly_dose("partial", Pf); doses[("partial", Pf)] = D; m = fit(D)
    res[f"hourly dose, partial cleaning (fraction P/{Pf} mm)"] = dict(aic=m.aic, b=m.params.soil, se=m.bse.soil, p=m.pvalues.soil)
D_none = hourly_dose("full", 1e9); m = fit(D_none)
res["hourly dose, no rain cleaning (washes only)"] = dict(aic=m.aic, b=m.params.soil, se=m.bse.soil, p=m.pvalues.soil)
m_c = fit(doses[("full", 1)], extra={"aod_cams": L.aod_cams}); res["P >= 1 mm, + hourly CAMS AOD control"] = dict(aic=m_c.aic, b=m_c.params.soil, se=m_c.bse.soil, p=m_c.pvalues.soil)
aic0 = res["no soiling term"]["aic"]
for k_, v in res.items():
    v["dAIC_vs_no_soiling"] = round(v["aic"] - aic0, 1)
    if "b" in v: v["pct_per_dose_unit"] = round(100 * v["b"] / MR, 2)
T1 = pd.DataFrame(res).T.round(5); T1.to_csv(f"{OUT}/N6_hourly_reset_rules.csv")
R["H1_n_clear_zone_hours"] = int(len(L)); R["H1_n_days"] = int(L.date.nunique()); R["H1_models"] = T1.to_dict(orient="index")
best_key = min([k_ for k_ in res if k_.startswith("hourly dose, event") or k_.startswith("hourly dose, partial")], key=lambda k_: res[k_]["aic"])
R["H1_best"] = best_key

# ---------------- H2 event study ----------------
bkey = ("full", float(best_key.split(">= ")[1].split(" ")[0])) if "event reset" in best_key else ("partial", float(best_key.split("/")[1].split(" ")[0]))
Dbest = doses[bkey]
if CAMS_CTRL: base = base.assign(aod_cams=L.aod_cams.values)
mb = fit(Dbest); B_S = mb.params.soil
Xb = base.assign(soil=Dbest.reindex(L.index).values)
L["res"] = L.ratio - (Xb.drop(columns="soil") * mb.params.drop("soil")).sum(1)      # soiling effect left in the residual
r_h = L.groupby(L.index).res.mean() / MR * 100                                      # fleet residual, % of mean ratio
Dh = Dbest
def window_stats(t0, t1):
    b = r_h[(r_h.index >= t0 - pd.Timedelta(hours=72)) & (r_h.index < t0)]
    a = r_h[(r_h.index > t1) & (r_h.index <= t1 + pd.Timedelta(hours=72))]
    if len(b) < 3 or len(a) < 3: return None
    return dict(n_b=len(b), n_a=len(a), obs=a.mean() - b.mean(), se=np.sqrt(a.var() / len(a) + b.var() / len(b)),
                dAOD_cams=float(H.aod_cams.reindex(a.index).mean() - H.aod_cams.reindex(b.index).mean()),
                dose_pre=float(Dh.asof(t0 - pd.Timedelta(hours=1))), pred=float(100 * -B_S * (Dh.reindex(b.index).mean() - Dh.reindex(a.index).mean()) / MR))
ev = []
for eid_, e in EV.iterrows():
    if not (e.dry_before_h >= 48): continue
    s = window_stats(e.t0, e.t1)
    if s: ev.append(dict(t0=e.t0, P=round(e.P, 1), dry_h=e.dry_before_h, **s))
EVS = pd.DataFrame(ev); EVS.round(3).to_csv(f"{OUT}/N6_rain_events_hourly{TAG}.csv", index=False)
# placebo: pseudo-events at 14:00 on days inside dry spells (no rain within +/- 96 h)
dry = H.prcp.rolling(193, center=True, min_periods=1).sum() == 0
pl = []
for t0 in H.index[(H.index.hour == 14) & dry.values][::3]:
    s = window_stats(t0, t0)
    if s: pl.append(dict(t0=t0, **s))
PL = pd.DataFrame(pl)
def summ(df):
    if len(df) < 4: return dict(n=int(len(df)))
    w = 1 / df.se.clip(lower=0.5) ** 2
    f1 = sm.WLS(df.obs, sm.add_constant(df.dose_pre), weights=w).fit(cov_type="HC1")
    f2 = sm.WLS(df.obs, sm.add_constant(df.pred), weights=w).fit(cov_type="HC1")
    return dict(n=int(len(df)), mean_obs=round(float(np.average(df.obs, weights=w)), 2), se_mean=round(float(1 / np.sqrt(w.sum())), 2),
                mean_dAOD_cams=round(float(df.dAOD_cams.mean()), 3),
                mean_pred=round(float(df.pred.mean()), 2), slope_obs_per_dose=round(f1.params.dose_pre, 2), slope_dose_se=round(f1.bse.dose_pre, 2),
                slope_dose_p=round(f1.pvalues.dose_pre, 3), slope_obs_on_pred=round(f2.params.pred, 2), slope_pred_se=round(f2.bse.pred, 2))
S = {"all events (dry >= 48 h before)": summ(EVS), "placebo dry periods": summ(PL)}
for lab, lo, hi in (("0.1-1 mm", 0.1, 1), ("1-2 mm", 1, 2), ("2-5 mm", 2, 5), (">= 5 mm", 5, 1e9)):
    S[f"events {lab}"] = summ(EVS[(EVS.P >= lo) & (EVS.P < hi)])
hi_d = EVS.dose_pre >= EVS.dose_pre.median()
S["events with pre-rain dose >= median"] = summ(EVS[hi_d]); S["events with pre-rain dose < median"] = summ(EVS[~hi_d])
for lab, lo, hi in (("< 1 mm", 0.1, 1), (">= 1 mm", 1, 1e9)):
    S[f"high pre-dose events {lab}"] = summ(EVS[hi_d & (EVS.P >= lo) & (EVS.P < hi)])
R["H2_best_rule"] = str(bkey); R["H2_B_S"] = float(B_S); R["H2_B_S_pct"] = float(100 * B_S / MR); R["H2_event_study"] = S
json.dump(R, open(f"{OUT}/N6_rain_hourly_results{TAG}.json", "w"), indent=1, default=str)
print(json.dumps(R, indent=1, default=str))

