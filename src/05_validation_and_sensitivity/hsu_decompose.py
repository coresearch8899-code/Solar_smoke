"""Which part of the HSU model improves the fit: the rain threshold, the coarse (PM10 - PM2.5) fraction, or the erf shape?
Linear dose regressors built from PM2.5, PM10, the coarse fraction and the HSU deposition-velocity weighting, each with 0.5-5 mm rain resets,
in the same clear-day PV model as hsu_benchmark.py. Also checks whether the soiling regressor is confounded with the season (adds a
first-order seasonal harmonic) and reports the Jan-May mean modelled deposition loss implied by each fit (% of the mean ratio)."""
import sys, json, numpy as np, pandas as pd, statsmodels.api as sm
D3, OUT = sys.argv[1:3]
KWP = {"Z1": 2256.24, "Z2": 207.06, "Z3": 49.90}
CLEAN = pd.to_datetime(["2023-05-06", "2024-03-09", "2024-06-23", "2025-04-25", "2025-12-26", "2026-03-23", "2026-09-05"])
COMMISSION = pd.Timestamp("2023-03-03")
d = pd.read_csv(D3, index_col=0, parse_dates=True); d = d[d.index >= COMMISSION]
pm25, pm10 = d.pm25.fillna(0), d.pm10.fillna(0); coarse = np.maximum(pm10 - pm25, 0)
SRC = {"PM2.5": pm25, "PM10": pm10, "coarse": coarse, "HSU-weighted": pm25 * 0.0009 / 0.004 + coarse}
def dose(x, thr):
    reset = (d.e5_tp >= thr) | d.index.isin(CLEAN + pd.Timedelta(days=1)) | (d.index == COMMISSION)
    return x.groupby(reset.cumsum()).cumsum() / 1000.0
doy = d.index.dayofyear
def frame(soil):
    out = []
    for z in KWP:
        t = pd.DataFrame({"ratio": d[f"Y_{z}"] / d.e5_ssrd, "aod": d.aod_ceres, "t2m": d.e5_t2m, "soil": soil,
                          "s1": np.sin(2 * np.pi * doy / 365.25), "c1": np.cos(2 * np.pi * doy / 365.25),
                          "clear": (d.e5_tcc_day < 30) & (d.ceres_all / d.ceres_clr > 0.85) & (d.e5_tp < 1),
                          "yr": (d.index - COMMISSION).days / 365.25, "zone": z}, index=d.index)
        med = t.ratio.median(); out.append(t[t.clear & t.ratio.between(0.4 * med, 1.3 * med)])
    return pd.concat(out).drop(columns="clear").dropna()
def fit(P, season=False):
    X = pd.get_dummies(P.zone).astype(float)
    for z in KWP: X[f"tr_{z}"] = (P.zone == z) * P.yr
    cols = ["aod", "t2m", "soil"] + (["s1", "c1"] if season else [])
    X[cols] = P[cols]; return sm.OLS(P.ratio, X).fit()
jm = d.index.month.isin([1, 2, 3, 4, 5]); rows = []
for src, x in SRC.items():
    for thr in (0.5, 1.0, 2.0, 5.0):
        s = dose(x, thr); P = frame(s); MR = P.ratio.mean()
        for season in (False, True):
            m = fit(P, season)
            rows.append(dict(source=src, reset_mm=thr, season_harmonic=season, aic=round(m.aic, 1), coef=m.params.soil, p=m.pvalues.soil,
                             JanMay_loss_pct=round(100 * float(-m.params.soil * s[jm].mean() / MR), 2)))
from pvlib.soiling import hsu                                      # HSU soiling loss 1 - SR as the regressor (tilt 10 deg assumed)
rain = d.e5_tp.fillna(0).copy(); rain[rain.index.isin(CLEAN)] = 1000.0
for thr in (1.0, 2.0, 5.0):
    s = 1 - hsu(rain, cleaning_threshold=thr, surface_tilt=10.0, pm2_5=pm25 * 1e-6, pm10=pm10 * 1e-6, rain_accum_period=pd.Timedelta("1D"))
    P = frame(s); MR = P.ratio.mean()
    for season in (False, True):
        m = fit(P, season)
        rows.append(dict(source="HSU 1-SR", reset_mm=thr, season_harmonic=season, aic=round(m.aic, 1), coef=m.params.soil, p=m.pvalues.soil,
                         JanMay_loss_pct=round(100 * float(-m.params.soil * s[jm].mean() / MR), 2)))
R = pd.DataFrame(rows); base = R[(R.source == "PM2.5") & (R.reset_mm == 2.0)].set_index("season_harmonic").aic
R["dAIC_vs_paper"] = R.apply(lambda r: round(r.aic - base[r.season_harmonic], 1), axis=1)
R.to_csv(f"{OUT}/hsu_decompose.csv", index=False); print(R.round(4).to_string(index=False))
