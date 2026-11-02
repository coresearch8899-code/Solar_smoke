"""Step 1 – external-data analysis for the haze/fire–PV manuscript (CRH, Chiang Rai).
Inputs: Open-Meteo CAMS air quality + ERA5 (hourly), NASA POWER (CERES/MERRA-2 daily),
VIIRS FIRMS archives (SNPP, NOAA-20, NOAA-21), GADM 4.1, NOAA ONI, Air4Thai 57T (Aug-Sep 2026).
"""
import sys, io, json
import numpy as np, pandas as pd, geopandas as gpd
import statsmodels.api as sm
import pvlib

D = sys.argv[1]; OUT = sys.argv[2]
LAT, LON, KWP, PR = 19.905, 99.83, 2513.2, 0.80

# ---------- 1. hourly CAMS + ERA5 -> daily ----------
aq = pd.read_csv(f"{D}/open-meteo-19.90N99.80E398m.csv", skiprows=3)
aq.columns = ["time", "pm25", "pm10", "co", "aod", "dust"]
met = pd.read_csv(f"{D}/open-meteo-20.00N99.75E398m.csv", skiprows=3)
met.columns = ["time", "t2m", "rh", "prcp", "tcc", "sw", "dni_h", "dhi", "ws10", "wd10"]
for x in (aq, met):
    x["time"] = pd.to_datetime(x["time"]); x["date"] = x["time"].dt.normalize()
day = (met.assign(day_tcc=np.where(met.sw > 0, met.tcc, np.nan))
          .groupby("date").agg(t2m=("t2m", "mean"), rh=("rh", "mean"), prcp=("prcp", "sum"),
                               tcc=("day_tcc", "mean"), era5_ghi=("sw", lambda s: s.sum() / 1000),
                               ws10=("ws10", "mean")))
aqd = aq.groupby("date").agg(pm25=("pm25", "mean"), pm10=("pm10", "mean"), co=("co", "mean"),
                             aod_cams=("aod", "mean"))
pw = pd.read_csv(f"{D}/POWER_Point_Daily_20230301_20260930_019d91N_099d83E_LST.csv", skiprows=15)
pw["date"] = pd.to_datetime(dict(year=pw.YEAR, month=pw.MO, day=pw.DY))
pw = pw.set_index("date")[["ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN", "AOD_55"]].replace(-999, np.nan)
pw.columns = ["ceres_all", "ceres_clr", "aod_ceres"]
df = day.join(aqd).join(pw)
df = df.loc["2023-03-01":"2026-09-30"]

# extraterrestrial daily horizontal irradiation (kWh/m2)
idx = pd.date_range("2023-03-01", "2026-09-30 23:00", freq="10min", tz="Asia/Bangkok")
sp = pvlib.solarposition.get_solarposition(idx, LAT, LON)
etr = pvlib.irradiance.get_extra_radiation(idx) * np.clip(np.cos(np.radians(sp.zenith)), 0, None)
etr_d = (etr.resample("D").sum() / 6 / 1000); etr_d.index = etr_d.index.tz_localize(None)
df["etr"] = etr_d
df["tau_clr"] = df.ceres_clr / df.etr           # clear-sky transmittance (CERES, real aerosols)

# ---------- 2. fires ----------
fr = []
for tag, folder in [("SNPP", "DL_FIRE_SV-C2_823148/fire_%s_SV-C2_823148.csv"),
                    ("N20", "DL_FIRE_J1V-C2_823146/fire_%s_J1V-C2_823146.csv"),
                    ("N21", "DL_FIRE_J2V-C2_823147/fire_%s_J2V-C2_823147.csv")]:
    for kind in ("archive", "nrt"):
        f = pd.read_csv(f"{D}/" + folder % kind)
        f["sat"] = tag; fr.append(f)
fr = pd.concat(fr, ignore_index=True)
if "type" in fr: fr = fr[(fr["type"].isna()) | (fr["type"] == 0)]
fr = fr[fr.confidence.isin(["n", "h"])]
t_utc = pd.to_datetime(fr.acq_date) + pd.to_timedelta(fr.acq_time // 100, "h") + pd.to_timedelta(fr.acq_time % 100, "m")
fr["date"] = (t_utc + pd.Timedelta(hours=7)).dt.normalize()
g = gpd.GeoDataFrame(fr, geometry=gpd.points_from_xy(fr.longitude, fr.latitude), crs=4326)
adm = pd.concat([gpd.read_file(f"{D}/gadm41_{c}.gpkg", layer="ADM_ADM_1")[["GID_0", "NAME_1", "geometry"]]
                 for c in ("THA", "MMR", "LAO")])
g = gpd.sjoin(g, adm, how="left", predicate="within").drop(columns="index_right")
R = 6371.0
la1, lo1 = np.radians(LAT), np.radians(LON)
la2, lo2 = np.radians(g.latitude), np.radians(g.longitude)
g["dist_km"] = 2 * R * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))
UPPER_N = {"Chiang Mai", "Mae Hong Son", "Lamphun", "Lampang", "Phayao", "Phrae", "Nan", "Uttaradit", "Tak"}
g["region"] = np.select([g.NAME_1.eq("Chiang Rai"), g.NAME_1.isin(UPPER_N), g.GID_0.eq("THA")],
                        ["CR", "UpperN", "RestTH"], "Other")
country_counts = g.GID_0.fillna("none").value_counts().to_dict()
# daily FRP per satellite then mean across satellites available that day (robust to N21 start)
def daily_frp(mask, name):
    s = g[mask].groupby(["date", "sat"]).frp.sum().unstack(fill_value=0)
    nsat = g.groupby(["date", "sat"]).size().unstack().notna().sum(axis=1)
    return (s.sum(axis=1) / nsat.reindex(s.index).clip(lower=1)).rename(name)
frp = pd.concat([daily_frp(g.region.eq(r), f"frp_{r}") for r in ("CR", "UpperN", "RestTH")] +
                [daily_frp(g.dist_km < 100, "frp_lt100"), daily_frp(g.dist_km.between(100, 300), "frp_100_300"),
                 daily_frp(g.dist_km > 300, "frp_gt300")], axis=1)
df = df.join(frp).fillna({c: 0 for c in frp.columns})

# ---------- 3. ONI ----------
oni_txt = """DJF 2023 -0.55|JFM 2023 -0.33|FMA 2023 -0.11|MAM 2023 0.19|AMJ 2023 0.46|MJJ 2023 0.73|JJA 2023 1.00|JAS 2023 1.25|ASO 2023 1.50|SON 2023 1.74|OND 2023 1.90|NDJ 2023 1.99|DJF 2024 1.84|JFM 2024 1.53|FMA 2024 1.18|MAM 2024 0.77|AMJ 2024 0.43|MJJ 2024 0.18|JJA 2024 0.06|JAS 2024 -0.04|ASO 2024 -0.12|SON 2024 -0.19|OND 2024 -0.29|NDJ 2024 -0.43|DJF 2025 -0.46|JFM 2025 -0.22|FMA 2025 -0.08|MAM 2025 0.02|AMJ 2025 -0.04|MJJ 2025 -0.02|JJA 2025 -0.11|JAS 2025 -0.26|ASO 2025 -0.43|SON 2025 -0.57|OND 2025 -0.61|NDJ 2025 -0.60|DJF 2026 -0.39|JFM 2026 -0.21|FMA 2026 0.11|MAM 2026 0.46|AMJ 2026 0.95|MJJ 2026 1.39|JJA 2026 1.80|JAS 2026 2.16"""
oni = {}
for i, tok in enumerate(oni_txt.split("|")):
    s, y, v = tok.split(); oni[(int(y), i % 12 + 1)] = float(v)   # centre month: DJF->Jan
df["oni"] = [oni.get((d.year, d.month), np.nan) for d in df.index]

# ---------- 4. Air4Thai 57T vs CAMS (only Aug-Sep 2026 served by the API) ----------
a4t = """2026-08-01,10.3|2026-08-02,10.6|2026-08-03,10.9|2026-08-04,10.1|2026-08-05,10.0|2026-08-06,8.4|2026-08-07,6.1|2026-08-08,4.8|2026-08-09,5.2|2026-08-10,4.5|2026-08-11,5.2|2026-08-12,7.1|2026-08-13,8.0|2026-08-14,9.1|2026-08-15,8.0|2026-08-16,6.9|2026-08-17,10.4|2026-08-18,14.8|2026-08-19,12.6|2026-08-20,8.5|2026-08-21,8.1|2026-08-22,7.5|2026-08-23,9.7|2026-08-24,11.4|2026-08-25,13.6|2026-08-26,10.0|2026-08-27,7.5|2026-08-28,7.6|2026-08-29,8.7|2026-08-30,7.5|2026-08-31,7.1|2026-09-01,9.0|2026-09-02,11.2|2026-09-03,10.9|2026-09-04,7.5|2026-09-05,5.7|2026-09-06,8.5|2026-09-07,9.5|2026-09-08,8.5|2026-09-09,9.8|2026-09-10,6.8|2026-09-11,12.0|2026-09-12,15.7|2026-09-13,14.4|2026-09-14,16.0|2026-09-15,16.0|2026-09-16,10.0|2026-09-17,8.6|2026-09-18,7.0|2026-09-19,6.2|2026-09-20,6.9|2026-09-21,7.4|2026-09-22,7.1|2026-09-23,14.6|2026-09-24,13.3|2026-09-25,14.7|2026-09-26,10.5|2026-09-27,7.0|2026-09-28,7.3|2026-09-29,6.6|2026-09-30,8.9"""
a = pd.Series({pd.Timestamp(k): float(v) for k, v in (t.split(",") for t in a4t.split("|"))})
df["pm25_57T"] = a
v = df[["pm25", "pm25_57T"]].dropna()
val = dict(n=len(v), r=v.corr().iloc[0, 1], bias=(v.pm25 - v.pm25_57T).mean(), mean_obs=v.pm25_57T.mean())

# ---------- 5. aerosol attenuation of clear-sky irradiance ----------
c = df[["tau_clr", "aod_ceres"]].dropna()
X = sm.add_constant(c.aod_ceres); m_tau = sm.OLS(c.tau_clr, X).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
b0, b1 = m_tau.params
# background AOD = median monsoon (Jun-Sep) AOD; loss vs background, scaled to all-sky by ALL/CLR ratio
aod_bg = df.loc[df.index.month.isin([11, 12, 1]), "aod_ceres"].median()   # clean dry season (Nov-Jan)
aod_bg_monsoon = df.loc[df.index.month.isin([6, 7, 8, 9]), "aod_ceres"].median()
df["dH_clr"] = -b1 * (df.aod_ceres - aod_bg).clip(lower=0) * df.etr        # kWh/m2/d
df["dH_all"] = df.dH_clr * (df.ceres_all / df.ceres_clr).clip(0, 1)
df["dE_fleet_kWh"] = df.dH_all * KWP * PR                                    # indicative
df["E_fleet_kWh"] = df.ceres_all * KWP * PR
# reanalysis haze bias on clear days (ERA5 uses aerosol climatology)
clr = df[(df.tcc < 20)].dropna(subset=["ceres_all", "era5_ghi", "aod_ceres"])
m_bias = sm.OLS(clr.ceres_all / clr.era5_ghi, sm.add_constant(clr.aod_ceres)).fit(cov_type="HAC", cov_kwds={"maxlags": 7})

# ---------- 6. fires -> PM2.5 (CAMS) attribution, haze season ----------
H = df[df.index.month.isin([1, 2, 3, 4, 5])].copy()
regs = ["frp_CR", "frp_UpperN", "frp_RestTH"]
for r in regs:
    for L in (0, 1, 2):
        H[f"l{r}_{L}"] = np.log1p(df[r].shift(L)).reindex(H.index)
H["lag_pm"] = np.log(df.pm25.shift(1)).reindex(H.index)
xcols = [f"l{r}_{L}" for r in regs for L in (0, 1, 2)] + ["rh", "ws10", "prcp", "t2m"]
Hm = H.dropna(subset=xcols + ["pm25"])
m_pm = sm.OLS(np.log(Hm.pm25), sm.add_constant(Hm[xcols])).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
yhat = m_pm.predict(sm.add_constant(Hm[xcols]))
attr = {}
for r in regs:
    Z = Hm[xcols].copy(); Z[[f"l{r}_{L}" for L in (0, 1, 2)]] = 0
    attr[r] = float((np.exp(yhat) - np.exp(m_pm.predict(sm.add_constant(Z, has_constant="add")))).mean())
Z = Hm[xcols].copy(); Z[[c for c in xcols if c.startswith("lfrp")]] = 0
attr["all_TH_fires"] = float((np.exp(yhat) - np.exp(m_pm.predict(sm.add_constant(Z, has_constant="add")))).mean())

# ---------- 7. seasonal summaries ----------
df["year"] = df.index.year
hs = df[df.index.month.isin([3, 4])]                # Mar-Apr: comparable across all 4 years
season = hs.groupby("year").agg(pm25=("pm25", "mean"), days_gt37_5=("pm25", lambda s: int((s > 37.5).sum())),
                                aod=("aod_ceres", "mean"), frp_CR=("frp_CR", "sum"), frp_UpperN=("frp_UpperN", "sum"),
                                frp_RestTH=("frp_RestTH", "sum"), max_day_dE=("dE_fleet_kWh", "max"), dH_all=("dH_all", "sum"),
                                dE_MWh=("dE_fleet_kWh", lambda s: s.sum() / 1000), oni=("oni", "mean"))
jan_may = df[df.index.month.isin([1, 2, 3, 4, 5])].groupby("year").agg(
    dE_MWh=("dE_fleet_kWh", lambda s: s.sum() / 1000), pm25=("pm25", "mean"), n=("pm25", "count"))
annual_loss = df.groupby("year").agg(dE_MWh=("dE_fleet_kWh", lambda s: s.sum() / 1000), E_MWh=("E_fleet_kWh", lambda s: s.sum() / 1000), n=("pm25", "count"))
annual_loss["loss_pct"] = 100 * annual_loss.dE_MWh / (annual_loss.E_MWh + annual_loss.dE_MWh)
monthly = df.groupby(df.index.month).agg(pm25=("pm25", "mean"), aod=("aod_ceres", "mean"),
                                         frp_TH=("frp_RestTH", "mean"), dH=("dH_all", "mean"))
corr = df[["pm25", "co", "aod_cams", "aod_ceres", "tau_clr"]].corr().round(2)

res = dict(country_counts=country_counts, n_fire_points=int(len(g)),
           fire_bbox=[float(g.longitude.min()), float(g.latitude.min()), float(g.longitude.max()), float(g.latitude.max())],
           validation_57T=val, tau_model=dict(b0=b0, b1=b1, se=m_tau.bse.iloc[1], r2=m_tau.rsquared, n=int(m_tau.nobs)),
           aod_bg=aod_bg, aod_bg_monsoon=aod_bg_monsoon, era5_bias=dict(b0=m_bias.params.iloc[0], b1=m_bias.params.iloc[1], se=m_bias.bse.iloc[1],
                                         n=int(m_bias.nobs), r2=m_bias.rsquared),
           pm_model=dict(r2=m_pm.rsquared, n=int(m_pm.nobs)), pm_attr_ugm3=attr,
           mean_haze_pm=float(Hm.pm25.mean()))
print(json.dumps(res, indent=1, default=float))
print(m_pm.summary().tables[1])
print("\nMar-Apr season\n", season.round(2).to_string())
print("\nJan-May loss\n", jan_may.round(1).to_string())
print("\nAnnual loss\n", annual_loss.round(1).to_string())
print("\nMonthly climatology\n", monthly.round(3).to_string())
print("\nCorr\n", corr.to_string())
df.round(4).to_csv(f"{OUT}/CRH_haze_external_daily.csv")
g.groupby(["GID_0", "NAME_1"], dropna=False).agg(n=("frp", "size"), frp=("frp", "sum"),
     dist=("dist_km", "median")).sort_values("frp", ascending=False).to_csv(f"{OUT}/fire_by_province.csv")
