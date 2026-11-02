"""rev25 robustness analyses requested in the pre-submission review (all on the step-3 daily table, same PV frame as step 4/5).
A  soiling-metric comparison: AIC, leave-one-year-out (LOYO) RMSE, 2024-2025 deposition loss and break-even wash cost per metric,
   with and without a seasonal harmonic; includes a daily partial-cleaning dose (removal fraction min(1, P/2 mm)).
B  collinearity of the soiling dose with AOD, temperature and season on the clear days (correlations, VIF).
C  temporal validation of the PV model (LOYO by calendar year) and residual autocorrelation.
D  block-length sensitivity of the PV bootstrap (7, 14, 28-day blocks; zones of a day resampled together).
E  washing replay: break-even by soiling metric, partial-cleaning rain rule, and a train (2023-2024) / test (2025-2026) split.
F  share of the modelled losses that falls on clear days.
Usage: robustness_rev25.py <CRH_haze_daily_step3.csv> <out_dir>"""
import sys, json, warnings, numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
warnings.filterwarnings("ignore")
D3, OUT = sys.argv[1:3]
KWP = {"Z1": 2256.24, "Z2": 207.06, "Z3": 49.90}; KWP_TOT = sum(KWP.values())
CLEAN = pd.to_datetime(["2023-05-06", "2024-03-09", "2024-06-23", "2025-04-25", "2025-12-26", "2026-03-23", "2026-09-05"])
COMMISSION = pd.Timestamp("2023-03-03"); TARIFF = 4.2
d = pd.read_csv(D3, index_col=0, parse_dates=True)
AOD_BG = d.loc[d.index.month.isin([11, 12, 1]), "aod_ceres"].median()
CLRW = (d.ceres_all / d.ceres_clr).clip(0, 1)
pm25, pm10 = d.pm25.fillna(0), d.pm10.fillna(0); coarse = np.maximum(pm10 - pm25, 0)
SRC = {"PM2.5": pm25, "coarse": coarse, "HSU-weighted": pm25 * 0.0009 / 0.004 + coarse}
WASH = d.index.isin(CLEAN + pd.Timedelta(days=1)) | (d.index == COMMISSION)
def dose(x, thr):
    reset = (d.e5_tp >= thr) | WASH
    return x.groupby(reset.cumsum()).cumsum() / 1000.0
def dose_partial(x, pf=2.0):
    """Daily analogue of the hourly partial-cleaning model: D_t = D_{t-1} (1 - min(1, P_t/pf)) + x_t, full reset by washes."""
    out = np.zeros(len(d)); D = 0.0; P = d.e5_tp.fillna(0).values; xv = x.values / 1000.0; w = WASH
    for i in range(len(d)):
        D = 0.0 if w[i] else D * (1 - min(1.0, P[i] / pf)); D += xv[i]; out[i] = D
    return pd.Series(out, index=d.index)
def elapsed(thr):
    reset = (d.e5_tp >= thr) | WASH; one = pd.Series(1.0, index=d.index)
    return one.groupby(reset.cumsum()).cumsum()
METRICS = {  # name: (series, kind, reset rule for the replay)
    "PM2.5 dose, partial rain cleaning (primary)": (dose_partial(pm25), ("partial", pm25, 2.0)),
    "PM2.5 dose, partial rain cleaning, complete at 5 mm": (dose_partial(pm25, 5.0), ("partial", pm25, 5.0)),
    "PM2.5 dose, 2 mm reset (rev24 primary)": (dose(pm25, 2.0), ("reset", pm25, 2.0)),
    "PM2.5 dose, 0.5 mm reset": (dose(pm25, 0.5), ("reset", pm25, 0.5)),
    "PM2.5 dose, 5 mm reset": (dose(pm25, 5.0), ("reset", pm25, 5.0)),
    "Coarse dose, 2 mm reset": (dose(coarse, 2.0), ("reset", coarse, 2.0)),
    "Coarse dose, 0.5 mm reset": (dose(coarse, 0.5), ("reset", coarse, 0.5)),
    "HSU-weighted dose, 2 mm reset": (dose(SRC["HSU-weighted"], 2.0), ("reset", SRC["HSU-weighted"], 2.0)),
    "HSU-weighted dose, 0.5 mm reset": (dose(SRC["HSU-weighted"], 0.5), ("reset", SRC["HSU-weighted"], 0.5)),
    "Days since reset, 20 mm": (elapsed(20.0), None),
}
doy = d.index.dayofyear; S1, C1 = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
def frame(soil):
    out = []
    for z in KWP:
        t = pd.DataFrame({"ratio": d[f"Y_{z}"] / d.e5_ssrd, "aod": d.aod_ceres, "t2m": d.e5_t2m, "soil": soil, "s1": S1, "c1": C1,
                          "clear": (d.e5_tcc_day < 30) & (d.ceres_all / d.ceres_clr > 0.85) & (d.e5_tp < 1),
                          "yr": (d.index - COMMISSION).days / 365.25, "zone": z}, index=d.index)
        med = t.ratio.median(); out.append(t[t.clear & t.ratio.between(0.4 * med, 1.3 * med)])
    P = pd.concat(out).drop(columns="clear").dropna(); return P[P.index >= COMMISSION]
def X_(P, season=False, soil=True):
    X = pd.get_dummies(P.zone).astype(float)
    for z in KWP: X[f"tr_{z}"] = (P.zone == z) * P.yr
    cols = ["aod", "t2m"] + (["soil"] if soil else []) + (["s1", "c1"] if season else [])
    X[cols] = P[cols]; return X
def fit(P, season=False, soil=True): return sm.OLS(P.ratio, X_(P, season, soil)).fit()
PRIMARY = "PM2.5 dose, partial rain cleaning (primary)"
def lomo_rmse(P, season=False, soil=True):
    """Out-of-sample RMSE with whole calendar months held out in turn (interpolation, no trend extrapolation)."""
    key = P.index.to_period("M"); err = []
    for k in sorted(set(key)):
        tr, te = P[key != k], P[key == k]
        m = fit(tr, season, soil); Xt = X_(te, season, soil).reindex(columns=m.params.index, fill_value=0.0); err.append(te.ratio - m.predict(Xt))
    e = pd.concat(err); return float(np.sqrt((e ** 2).mean())), e
def loyo_rmse(P, season=False, soil=True):
    """Out-of-sample RMSE: fit on all other calendar years, predict the held-out year (zone trends extrapolated)."""
    err = []
    for y in sorted(set(P.index.year)):
        tr, te = P[P.index.year != y], P[P.index.year == y]
        m = fit(tr, season, soil); Xt = X_(te, season, soil).reindex(columns=m.params.index, fill_value=0.0)
        err.append(te.ratio - m.predict(Xt))
    e = pd.concat(err); return float(np.sqrt((e ** 2).mean())), e
E_idx = d.index[(sum(d[f"Y_{z}"] * k for z, k in KWP.items())).notna()]
H_all = d.e5_ssrd
def replay(beta, rule, policy, param=None, period=None):
    kind, x, thr = rule; ALL = d.index[(d.index >= COMMISSION) & (d.index <= E_idx.max())]   # rev25: dose on every day, loss on fleet days
    idx = ALL if period is None else ALL[(ALL >= period[0]) & (ALL <= period[1])]; has = idx.isin(E_idx) & d.pm25.reindex(idx).notna().values
    xv = x.reindex(idx).fillna(0).values / 1000; tp = d.e5_tp.reindex(idx).fillna(0).values; H = H_all.reindex(idx).fillna(0).values
    D = 0.0; loss = 0.0; n = 0; cl = set(CLEAN + pd.Timedelta(days=1))
    for i, day in enumerate(idx):
        if kind == "reset": D = 0.0 if tp[i] >= thr else D
        else: D *= (1 - min(1.0, tp[i] / thr))
        if (policy == "actual" and day in cl) or (policy == "dose" and D >= param): D = 0.0; n += 1
        D += xv[i]; loss += -beta * D * H[i] * KWP_TOT * has[i]
    return n, loss / 1000
def breakeven(beta, rule, s, period=None):
    """Highest wash cost (THB) at which some dose threshold still saves money relative to no washing."""
    _, l0 = replay(beta, rule, "none", period=period); best = 0.0; bestD = None
    grid = np.unique(np.r_[np.quantile(s[s > 0], np.linspace(0.30, 0.995, 40)), np.arange(0.25, 4.01, 0.25)])   # step-5 grid included
    for D in grid:
        n, l = replay(beta, rule, "dose", D, period)
        if n:
            c = (l0 - l) * 1000 * TARIFF / n
            if c > best: best, bestD = c, D
    return best, bestD, l0

R = {}
# ---------------- A: soiling-metric comparison ----------------
P0 = frame(METRICS[PRIMARY][0])
rm_nosoil, _ = loyo_rmse(P0, soil=False); rm_nosoil_s, _ = loyo_rmse(P0, season=True, soil=False)
mo_nosoil, _ = lomo_rmse(P0, soil=False); mo_nosoil_s, _ = lomo_rmse(P0, season=True, soil=False)
full_yrs = (E_idx.year == 2024) | (E_idx.year == 2025)
rows = []; base_aic = {}
for name, (s, rule) in METRICS.items():
    P = frame(s); MR = P.ratio.mean()
    for season in (False, True):
        m = fit(P, season); rm, _ = loyo_rmse(P, season); mo, _ = lomo_rmse(P, season)
        dep = (-m.params.soil * s.reindex(E_idx) * H_all.reindex(E_idx) * KWP_TOT)[full_yrs].sum() / 2 / 1000
        jm = s.index.month.isin([1, 2, 3, 4, 5]) & (s.index >= COMMISSION)
        row = dict(metric=name, season_harmonic=season, n=int(m.nobs), aic=m.aic, beta_soil=m.params.soil, p_soil=m.pvalues.soil,
                   loyo_rmse=rm, loyo_gain_vs_no_soil_pct=100 * (1 - rm / (rm_nosoil_s if season else rm_nosoil)),
                   lomo_rmse=mo, lomo_gain_vs_no_soil_pct=100 * (1 - mo / (mo_nosoil_s if season else mo_nosoil)),
                   JanMay_dep_pct_of_ratio=100 * float(-m.params.soil * s[jm].mean() / MR), dep_MWh_per_yr_2024_25=dep)
        if rule is not None and not season and m.params.soil < 0:
            be, bD, _ = breakeven(m.params.soil, rule, s); row["breakeven_THB"] = be
        rows.append(row)
A = pd.DataFrame(rows)
for season in (False, True):
    b = A[(A.metric == PRIMARY) & (A.season_harmonic == season)].aic.iloc[0]
    A.loc[A.season_harmonic == season, "dAIC_vs_primary"] = A.loc[A.season_harmonic == season, "aic"] - b
A.to_csv(f"{OUT}/A_soiling_metric_comparison.csv", index=False)
R["A_loyo_rmse_no_soil"] = dict(no_season=rm_nosoil, season=rm_nosoil_s); R["A_lomo_rmse_no_soil"] = dict(no_season=mo_nosoil, season=mo_nosoil_s)

# ---------------- B: collinearity on the clear days ----------------
P = P0.copy(); cor = P[["soil", "aod", "t2m", "s1", "c1"]].corr()
cor.to_csv(f"{OUT}/B_correlations.csv")
Xv = sm.add_constant(P[["aod", "t2m", "soil", "s1", "c1"]]); R["B_VIF"] = {c: float(variance_inflation_factor(Xv.values, i)) for i, c in enumerate(Xv.columns) if c != "const"}
Xv2 = sm.add_constant(P[["aod", "t2m", "soil"]]); R["B_VIF_no_season"] = {c: float(variance_inflation_factor(Xv2.values, i)) for i, c in enumerate(Xv2.columns) if c != "const"}
R["B_corr_soil"] = cor["soil"].round(3).to_dict()
R["B_harmonic_R2_of_dose"] = float(sm.OLS(P.soil, sm.add_constant(P[["s1", "c1"]])).fit().rsquared)
R["B_dose_by_month"] = d.assign(D=METRICS[PRIMARY][0])[d.index >= COMMISSION].groupby(lambda t: t.month).D.mean().round(3).to_dict()

# ---------------- C: temporal validation of the PV model ----------------
m0 = fit(P0); rm, e = loyo_rmse(P0); mo, eo = lomo_rmse(P0)
byyr = {int(y): dict(n=int((e.index.year == y).sum()), rmse=float(np.sqrt((e[e.index.year == y] ** 2).mean())),
                     r2_oos=float(1 - (e[e.index.year == y] ** 2).sum() / ((P0.ratio[P0.index.year == y] - P0.ratio[P0.index.year == y].mean()) ** 2).sum()))
        for y in sorted(set(e.index.year))}
res = m0.resid.groupby(level=0).mean().asfreq("D")
R["C_PV_LOYO"] = dict(rmse=rm, by_year=byyr, in_sample_rmse=float(np.sqrt(m0.mse_resid)), in_sample_r2=m0.rsquared)
R["C_PV_LOMO"] = dict(rmse=mo, r2_oos=float(1 - (eo ** 2).sum() / ((P0.ratio - P0.ratio.mean()) ** 2).sum()))
R["C_resid_lag1_autocorr_daily_mean"] = float(res.autocorr(1)); R["C_durbin_watson_daily_mean"] = float(durbin_watson(res.dropna()))
R["C_resid_lag7_autocorr"] = float(res.autocorr(7))

# ---------------- D: block-length sensitivity of the bootstrap ----------------
rng = np.random.default_rng(42)
E = pd.DataFrame(index=E_idx)
E["A1"] = -(d.aod_ceres - AOD_BG).clip(lower=0) * d.e5_ssrd * CLRW * KWP_TOT          # atm loss per unit (-beta_A), kWh
E["S1"] = -METRICS[PRIMARY][0] * d.e5_ssrd * KWP_TOT
fy = E[(E.index.year == 2024) | (E.index.year == 2025)].sum() / 2 / 1000                # MWh per (unit beta) per year
D_rows = []
for L in (7, 14, 28):
    blk = (P0.index - P0.index.min()).days // L; ub = np.unique(blk); groups = {k: P0[blk == k] for k in ub}; out = []
    for b in range(1000):
        Pb = pd.concat([groups[k] for k in rng.choice(ub, len(ub), replace=True)]); mb = fit(Pb)
        out.append(dict(bA=mb.params.aod, bS=mb.params.soil, tot=mb.params.aod * fy.A1 + mb.params.soil * fy.S1))
    O = pd.DataFrame(out)
    D_rows.append(dict(block_days=L, bA_lo=O.bA.quantile(.025), bA_hi=O.bA.quantile(.975), bS_lo=O.bS.quantile(.025), bS_hi=O.bS.quantile(.975),
                       tot_lo=O.tot.quantile(.025), tot_hi=O.tot.quantile(.975)))
DB = pd.DataFrame(D_rows); DB.to_csv(f"{OUT}/D_block_length.csv", index=False)
R["D_point_total_linear"] = float(m0.params.aod * fy.A1 + m0.params.soil * fy.S1)

# ---------------- E: replay, train/test split ----------------
cut = pd.Timestamp("2024-12-31"); tr = P0[P0.index <= cut]; mtr = fit(tr)
rule = METRICS[PRIMARY][1]; s = METRICS[PRIMARY][0]
TT = {}
for cost in (10_000, 20_000, 40_000):
    # choose D* on the training period with the training coefficient, evaluate on the test period
    _, l0tr = replay(mtr.params.soil, rule, "none", period=(COMMISSION, cut)); best = None
    for D in np.quantile(s[s > 0], np.linspace(0.30, 0.995, 40)):
        n, l = replay(mtr.params.soil, rule, "dose", D, (COMMISSION, cut)); c = n * cost + l * 1000 * TARIFF
        if best is None or c < best[0]: best = (c, D)
    Dstar = best[1]
    for beta_name, beta in (("train_beta", mtr.params.soil), ("full_beta", m0.params.soil)):
        n0, l0 = replay(beta, rule, "none", period=(cut + pd.Timedelta(days=1), E_idx.max()))
        n, l = replay(beta, rule, "dose", Dstar, (cut + pd.Timedelta(days=1), E_idx.max()))
        TT[f"{cost}_{beta_name}"] = dict(Dstar_train=float(Dstar), test_washes=n, test_saving_THB=float((l0 - l) * 1000 * TARIFF - n * cost))
R["E_train_test"] = TT
R["E_beta_train"] = float(mtr.params.soil); R["E_beta_full"] = float(m0.params.soil)
be_tr, _, _ = breakeven(mtr.params.soil, rule, s, (cut + pd.Timedelta(days=1), E_idx.max()))
be_full_test, _, _ = breakeven(m0.params.soil, rule, s, (cut + pd.Timedelta(days=1), E_idx.max()))
R["E_breakeven_test_period"] = dict(train_beta=be_tr, full_beta=be_full_test)

# ---------------- F: losses on clear days ----------------
clear = (d.e5_tcc_day < 30) & (d.ceres_all / d.ceres_clr > 0.85) & (d.e5_tp < 1)
ff = E[(E.index.year == 2024) | (E.index.year == 2025)]
cl = clear.reindex(ff.index).fillna(False)
R["F_clear_day_share"] = dict(days=float(cl.mean()), atm=float(ff.A1[cl].sum() / ff.A1.sum()), dep=float(ff.S1[cl].sum() / ff.S1.sum()))
R["F_atm_without_fcs_scaling_ratio"] = float((-(d.aod_ceres - AOD_BG).clip(lower=0) * d.e5_ssrd).reindex(ff.index).sum() / ff.A1.sum() * KWP_TOT)

json.dump(R, open(f"{OUT}/robustness_rev25.json", "w"), indent=1, default=float)
pd.set_option("display.width", 250)
print(A.round(4).to_string(index=False)); print(DB.round(4).to_string(index=False)); print(json.dumps(R, indent=1, default=float))
