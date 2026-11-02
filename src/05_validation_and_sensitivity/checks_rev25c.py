"""rev25 third-round checks (pre-submission review 3), partial-cleaning dose:
A  stability of the atmospheric / deposition split: beta_A-beta_S correlation across PV bootstrap replicates; separate fits by zone, by season
   (Jan-May, Jun-Dec) and without zone trends, each with the implied 2024-2025 atmospheric and deposition losses (fleet).
B  forward validation (train on earlier years, test the next year): RMSE, bias and R2 by test year, coefficients by fold, with and without
   zone trends, against a model without soiling.
C  all-sky scaling check: beta_A fitted on clear, partly cloudy and cloudy days (CERES all-sky/clear-sky ratio > 0.85, 0.6-0.85, < 0.6)
   compared with the f_cs-scaled clear-day coefficient; annual atmospheric loss with f_cs, without scaling and with class-specific scaling.
D  wash-event check summary with a 95 % CI (from R7).
Usage: checks_rev25c.py <step3_csv> <out25_dir> <out_dir>"""
import sys, json, warnings, numpy as np, pandas as pd, statsmodels.api as sm
warnings.filterwarnings("ignore")
D3, O25, OUT = sys.argv[1:4]
KWP = {"Z1": 2256.24, "Z2": 207.06, "Z3": 49.90}; KWP_TOT = sum(KWP.values())
CLEAN = pd.to_datetime(["2023-05-06", "2024-03-09", "2024-06-23", "2025-04-25", "2025-12-26", "2026-03-23", "2026-09-05"])
COMMISSION = pd.Timestamp("2023-03-03")
d = pd.read_csv(D3, index_col=0, parse_dates=True)
AOD_BG = d.loc[d.index.month.isin([11, 12, 1]), "aod_ceres"].median(); RAT = d.ceres_all / d.ceres_clr; CLRW = RAT.clip(0, 1)
WASH = d.index.isin(CLEAN + pd.Timedelta(days=1)) | (d.index == COMMISSION)
keep = 1 - np.minimum(1.0, d.e5_tp.fillna(0).values / 2.0); x = d.pm25.values / 1000; o = np.full(len(d), np.nan); Dd = 0.0
for i in range(len(d)):
    Dd = 0.0 if WASH[i] else Dd * keep[i]
    if not np.isnan(x[i]): Dd += x[i]; o[i] = Dd
d["dose"] = o
def frame(day_filter):
    rows = []
    for z in KWP:
        t = pd.DataFrame({"ratio": d[f"Y_{z}"] / d.e5_ssrd, "aod": d.aod_ceres, "t2m": d.e5_t2m, "soil": d.dose, "sel": day_filter,
                          "yr": (d.index - COMMISSION).days / 365.25, "zone": z}, index=d.index)
        med = t.ratio.median(); rows.append(t[t.sel & t.ratio.between(0.4 * med, 1.3 * med)])
    P = pd.concat(rows).drop(columns="sel").dropna(); return P[P.index >= COMMISSION]
CLEAR = (d.e5_tcc_day < 30) & (RAT > 0.85) & (d.e5_tp < 1)
def X_(P, trends=True, soil=True):
    X = pd.get_dummies(P.zone).astype(float)
    if trends:
        for z in sorted(set(P.zone)): X[f"tr_{z}"] = (P.zone == z) * P.yr
    X[["aod", "t2m"] + (["soil"] if soil else [])] = P[["aod", "t2m"] + (["soil"] if soil else [])]; return X
PV = frame(CLEAR); m0 = sm.OLS(PV.ratio, X_(PV)).fit()
E = d.index[(sum(d[f"Y_{z}"] * k for z, k in KWP.items())).notna()]; fy = E[(E.year == 2024) | (E.year == 2025)]
SA = float((((d.aod_ceres - AOD_BG).clip(lower=0) * d.e5_ssrd * CLRW * KWP_TOT).reindex(fy)).sum() / 2 / 1000)   # MWh per unit -beta_A
SS = float(((d.dose * d.e5_ssrd * KWP_TOT).reindex(fy)).sum() / 2 / 1000)
def losses(bA, bS): a, s = -bA * SA, -bS * SS; return dict(bA=bA, bS=bS, atm=a, dep=s, tot=a + s, atm_frac=100 * a / (a + s))
R = {"A_main": losses(m0.params.aod, m0.params.soil)}
# ---- A: split stability
pvb = pd.read_csv(f"{O25}/pv_bootstrap_replicates.csv"); R["A_corr_bA_bS_bootstrap"] = float(pvb.b_aod.corr(pvb.b_soil))
R["A_corr_aod_dose_clear_days"] = float(PV.aod.corr(PV.soil))
R["A_by_zone"] = {z: losses(*[sm.OLS(g.ratio, X_(g)).fit().params[k] for k in ("aod", "soil")]) for z, g in PV.groupby("zone")}
for z, g in PV.groupby("zone"): R["A_by_zone"][z]["n"] = int(len(g))
seas = {"Jan–May": PV.index.month.isin([1, 2, 3, 4, 5]), "Jun–Dec": ~PV.index.month.isin([1, 2, 3, 4, 5])}
R["A_by_season"] = {}
for k, m in seas.items():
    mm = sm.OLS(PV[m].ratio, X_(PV[m])).fit(); R["A_by_season"][k] = dict(bA=mm.params.aod, bA_se=mm.bse.aod, bS=mm.params.soil, bS_se=mm.bse.soil, n=int(m.sum()))
mt = sm.OLS(PV.ratio, X_(PV, trends=False)).fit(); R["A_no_trends"] = losses(mt.params.aod, mt.params.soil)
# ---- B: forward validation
rows = []
for y in (2024, 2025, 2026):
    tr, te = PV[PV.index.year < y], PV[PV.index.year == y]
    for trends in (True, False):
        for soil in (True, False):
            m = sm.OLS(tr.ratio, X_(tr, trends, soil)).fit(); e = te.ratio - m.predict(X_(te, trends, soil).reindex(columns=m.params.index, fill_value=0))
            rows.append(dict(test_year=y, train=f"2023–{y - 1}", trends=trends, soil=soil, n_test=len(te), rmse=float(np.sqrt((e ** 2).mean())),
                             bias=float(e.mean()), r2=float(1 - (e ** 2).sum() / ((te.ratio - te.ratio.mean()) ** 2).sum()),
                             bA=float(m.params.aod), bS=float(m.params.soil) if soil else np.nan))
FV = pd.DataFrame(rows); FV.to_csv(f"{OUT}/forward_validation.csv", index=False); R["B_forward"] = rows
# ---- C: all-sky scaling
noRain = d.e5_tp < 1
classes = {"clear (>0.85)": CLEAR, "partly cloudy (0.6–0.85)": noRain & RAT.between(0.6, 0.85), "cloudy (<0.6)": noRain & (RAT < 0.6)}
R["C_classes"] = {}
for k, f in classes.items():
    P = frame(f); m = sm.OLS(P.ratio, X_(P)).fit(); fcs = float(CLRW.reindex(P.index).mean())
    R["C_classes"][k] = dict(n=int(len(P)), bA=m.params.aod, bA_se=m.bse.aod, mean_fcs=fcs, bA_expected=float(m0.params.aod * fcs / CLRW.reindex(PV.index).mean()))
atm_none = float((((d.aod_ceres - AOD_BG).clip(lower=0) * d.e5_ssrd * KWP_TOT).reindex(fy)).sum() / 2 / 1000) * -m0.params.aod
cls = pd.Series("cloudy (<0.6)", index=d.index); cls[RAT.between(0.6, 0.85)] = "partly cloudy (0.6–0.85)"; cls[RAT > 0.85] = "clear (>0.85)"
bmap = cls.map({k: v["bA"] for k, v in R["C_classes"].items()})
atm_cls = float((-(bmap) * (d.aod_ceres - AOD_BG).clip(lower=0) * d.e5_ssrd * KWP_TOT).reindex(fy).sum() / 2 / 1000)
R["C_annual_atm"] = dict(fcs=R["A_main"]["atm"], none=atm_none, class_specific=atm_cls)
# ---- D: wash events
W = pd.read_csv(f"{O25}/R7_wash_event_validation.csv").dropna(subset=["observed_gain_pct"])
mo, se = W.observed_gain_pct.mean(), float(np.sqrt((W.se_pct ** 2).sum()) / len(W))
R["D_wash"] = dict(n=int(len(W)), mean_observed=mo, ci95=[mo - 1.96 * se, mo + 1.96 * se], mean_predicted=float(W.predicted_gain_pct.mean()),
                   r=float(np.corrcoef(W.observed_gain_pct, W.predicted_gain_pct)[0, 1]))
json.dump(R, open(f"{OUT}/checks_rev25c.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in R.items() if k != "B_forward"}, indent=1, default=float)); print(FV.round(4).to_string(index=False))
