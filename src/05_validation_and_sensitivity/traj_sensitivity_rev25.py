"""rev25: trajectory sensitivity to start time and transport level (pre-submission review 3, item 5), January-May 2023-2025.
Winds: hourly ERA5 (N26 S14 W92 E106; u/v at 850 hPa and 10 m, BLH). Kinematic 72-h back-trajectories with the Heun scheme of
trajectories_hourly.py, started daily at 01, 03, 06 and 09 UTC (08, 10, 13, 16 local time) from the site with three transport levels:
  850      isobaric 850 hPa wind
  10m      10-m wind
  blw      boundary-layer weighted wind, U = U10 + 0.5 min(1, BLH/1500 m) (U850 - U10), i.e. the mean of the 10-m and 850-hPa winds when
           the boundary layer is at least 1.5 km deep and close to the 10-m wind when it is shallow (BLH interpolated along the path)
Fires: NOAA-20 (J1) VIIRS regional archive, nominal/high confidence, presumed vegetation, Dec-May, country from GADM 4.1 (THA, MMR, LAO;
other = outside these three), FRP gridded daily at 0.5 deg by local date (as trajectories_hourly.py). The 06 UTC 850 hPa set-up is checked
against the original tfe_h850.csv. Attribution: the haze, PV (partial-cleaning dose) and Shapley machinery of step6_N1_rev25.py on the days
common to all set-ups. Usage: traj_sensitivity_rev25.py <data_dir> <step3_csv> <step3_dir> <step6_dir> <step6_N1_rev25.py> <out_dir>"""
import sys, glob, os, json, time, sqlite3, struct, h5py, numpy as np, pandas as pd, shapely
def gpkg_adm0(path):
    """Country polygon from a GADM GeoPackage (ADM_ADM_0) without GDAL: strip the GeoPackage binary header, read the WKB with shapely."""
    blob = sqlite3.connect(path).execute("select geom from ADM_ADM_0").fetchone()[0]
    flags = blob[3]; env = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(flags >> 1) & 0x07]
    return shapely.from_wkb(bytes(blob[8 + env:]))
DATA, D3, TFE_OLD, TFE_NEW, N1SRC, OUT = sys.argv[1:7]
os.makedirs(OUT, exist_ok=True); t0 = time.time()
LAT0, LON0 = 19.905, 99.83; G = 0.5; R = 6371e3; C = ["THA", "MMR", "LAO", "OTH"]
ERA = glob.glob(os.path.join(DATA, "ERA5_Hourly_2021_2025_N26_W92_S14_E106*", "ERA5_Hourly_2021_2025_N26_W92_S14_E106"))[0]
# ---------------- fires ----------------
fr = []
for f in glob.glob(os.path.join(DATA, "DL_FIRE_J1V-C2_823168", "fire_*.csv")):
    d = pd.read_csv(f, usecols=lambda c: c in ["latitude", "longitude", "acq_date", "acq_time", "confidence", "frp", "type"], dtype={"confidence": str})
    d = d[d.confidence.astype(str).isin(["n", "h"])]
    if "type" in d: d = d[d["type"] == 0]
    d = d[d.latitude.between(14, 26) & d.longitude.between(92, 106) & pd.to_datetime(d.acq_date).dt.month.isin([12, 1, 2, 3, 4, 5])]
    fr.append(d)
fires = pd.concat(fr, ignore_index=True)
fires = fires.reset_index(drop=True); cc = np.full(len(fires), "OTH", dtype=object)
for c in ("THA", "MMR", "LAO"):
    poly = gpkg_adm0(os.path.join(DATA, f"gadm41_{c}.gpkg")); shapely.prepare(poly)
    m = (cc == "OTH") & shapely.contains_xy(poly, fires.longitude.values, fires.latitude.values); cc[m] = c
fires["country"] = cc
t = pd.to_datetime(fires.acq_date) + pd.to_timedelta(fires.acq_time.astype(int) // 100, "h") + pd.to_timedelta(fires.acq_time.astype(int) % 100, "m")
fires["date"] = (t + pd.Timedelta(hours=7)).dt.normalize()
fires["iy"] = ((fires.latitude - 14) // G).astype(int).clip(0, 23); fires["ix"] = ((fires.longitude - 92) // G).astype(int).clip(0, 27)
print("fires", len(fires), fires.groupby("country").size().to_dict(), f"{time.time() - t0:.0f}s", flush=True)
days = pd.date_range("2022-12-01", "2025-12-31", freq="D"); dindex = {d_: i for i, d_ in enumerate(days)}
grid = np.zeros((4, len(days), 24, 28), "float32")
for ci, c in enumerate(C):
    s = fires[fires.country == c].groupby(["date", "iy", "ix"]).frp.sum().reset_index(); s = s[s.date.isin(dindex)]
    grid[ci, s.date.map(dindex).values, s.iy.values, s.ix.values] = s.frp.values
# ---------------- winds and trajectories ----------------
def load(var, months):
    arrs, times = [], []
    for ym in months:
        with h5py.File(f"{ERA}/{var}/era5_{var}_N26W92S14E106_{ym}.nc", "r") as h:
            arrs.append(h[var][:]); times.append(h["time"][:]); lat, lon = h["latitude"][:], h["longitude"][:]
    return np.concatenate(arrs), np.concatenate(times) / 3600.0, lat, lon
def interp(F, T, t, la, lo, lat, lon):
    it = np.clip(np.searchsorted(T, t) - 1, 0, len(T) - 2); ft = (t - T[it]) / (T[it + 1] - T[it])
    y = (lat[0] - la) / 0.25; x = (lo - lon[0]) / 0.25
    inside = np.isfinite(y) & np.isfinite(x) & (y >= 0) & (y <= len(lat) - 1.001) & (x >= 0) & (x <= len(lon) - 1.001)
    y = np.clip(np.nan_to_num(y), 0, len(lat) - 1.001); x = np.clip(np.nan_to_num(x), 0, len(lon) - 1.001)
    iy, ix = y.astype(int), x.astype(int); fy, fx = y - iy, x - ix; out = 0
    for dt_, wt in ((0, 1 - ft), (1, ft)):
        a = it + dt_
        out = out + wt * (F[a, iy, ix] * (1 - fy) * (1 - fx) + F[a, iy + 1, ix] * fy * (1 - fx) + F[a, iy, ix + 1] * (1 - fy) * fx + F[a, iy + 1, ix + 1] * fy * fx)
    return np.where(inside, out, np.nan)
def run(U, V, T, lat, lon, starts):
    n = len(starts); la = np.full(n, LAT0); lo = np.full(n, LON0); t = starts.copy(); tr = [np.stack([t, la, lo], 1)]
    for _ in range(72):
        dt = -3600.0
        u1 = interp(U, T, t, la, lo, lat, lon); v1 = interp(V, T, t, la, lo, lat, lon)
        la1 = la + np.degrees(v1 * dt / R); lo1 = lo + np.degrees(u1 * dt / (R * np.cos(np.radians(la))))
        u2 = interp(U, T, t - 1, la1, lo1, lat, lon); v2 = interp(V, T, t - 1, la1, lo1, lat, lon)
        la = la + np.degrees(0.5 * (v1 + v2) * dt / R); lo = lo + np.degrees(0.5 * (u1 + u2) * dt / (R * np.cos(np.radians(la))))
        t = t - 1; tr.append(np.stack([t, la, lo], 1))
    return np.stack(tr, 1)
def exposure(tr, start_dates):
    tt = pd.to_datetime((tr[:, :, 0] * 3600).ravel().astype("int64"), unit="s")
    di = pd.Series((tt + pd.Timedelta(hours=7)).normalize()).map(dindex).values.reshape(tr.shape[0], -1)
    iy = (tr[:, :, 1] - 14) // G; ix = (tr[:, :, 2] - 92) // G
    valid = ~np.isnan(tr[:, :, 1]) & (iy >= 0) & (iy < 24) & (ix >= 0) & (ix < 28) & ~pd.isna(di)
    out = pd.DataFrame(index=start_dates)
    for ci, c in enumerate(C):
        e = np.zeros(tr.shape[:2]); e[valid] = grid[ci, di[valid].astype(int), iy[valid].astype(int), ix[valid].astype(int)]
        out[f"tfe_{c}"] = e[:, 1:].sum(1); out[f"tfe24_{c}"] = e[:, 1:25].sum(1)
    out["frac_valid"] = valid[:, 1:].mean(1); return out
SETUPS = [(h, lev) for h in (1, 3, 6, 9) for lev in ("850", "10m", "blw")]
res = {s: [] for s in SETUPS}
for yr in (2023, 2024, 2025):
    months = [f"{yr - 1}12"] + [f"{yr}{m:02d}" for m in range(1, 6)]
    U8, T, lat, lon = load("u850", months); V8 = load("v850", months)[0]; U10 = load("10u", months)[0]; V10 = load("10v", months)[0]
    BLH = load("blh", months)[0]; w = 0.5 * np.minimum(1.0, BLH / 1500.0)
    winds = {"850": (U8, V8), "10m": (U10, V10), "blw": (U10 + w * (U8 - U10), V10 + w * (V8 - V10))}
    for h, lev in SETUPS:
        st = pd.date_range(f"{yr}-01-01 {h:02d}:00", f"{yr}-05-31 {h:02d}:00", freq="D")
        st_h = st.values.astype("datetime64[s]").astype("int64") / 3600.0; st_h = st_h[st_h - 72 >= T[0]]
        sd = pd.to_datetime(st_h * 3600, unit="s").normalize()
        res[(h, lev)].append(exposure(run(*winds[lev], T, lat, lon, st_h), sd))
    print(yr, "trajectories done", f"{time.time() - t0:.0f}s", flush=True)
TF = {}
for (h, lev), v in res.items():
    k = f"{h + 7:02d}LT_{lev}"; o = pd.concat(v); o.to_csv(f"{OUT}/tfe_{k}.csv"); TF[k] = o
# check against the original hourly 06 UTC 850 hPa exposure
orig = pd.read_csv(f"{TFE_NEW}/tfe_h850.csv", index_col=0, parse_dates=True); new = TF["13LT_850"]
jj = orig.join(new, lsuffix="_o", rsuffix="_n", how="inner"); jj = jj[jj.index.year >= 2023]
CHK = {c: dict(r_log=float(np.corrcoef(np.log1p(jj[f"tfe_{c}_o"]), np.log1p(jj[f"tfe_{c}_n"]))[0, 1]), ratio_sum=float(jj[f"tfe_{c}_n"].sum() / jj[f"tfe_{c}_o"].sum())) for c in C}
print("check vs original tfe_h850:", json.dumps(CHK), flush=True)
# ---------------- attribution with the step-6 machinery (partial-cleaning dose, commissioning restriction) ----------------
src = open(N1SRC, encoding="utf-8").read(); src = src[: src.index("# ================= N1")]
g = {"__name__": "n1lib"}; sys.argv = ["n1", D3, TFE_OLD, TFE_NEW, "-", "-", "-", OUT]; exec(compile(src, "n1lib", "exec"), g)
YRS = [2023, 2024, 2025]; common = None; frames = {}
for k, tf in TF.items():
    H, fire = g["haze_frame"](tf, YRS); frames[k] = (H, fire); common = H.index if common is None else common.intersection(H.index)
rows = []
for k, (H, fire) in frames.items():
    H = H.loc[H.index.intersection(common)]; v, mA, mP = g["engine"](H, fire); phi, joint = g["shapley"](v)
    ex = TF[k].loc[TF[k].index.isin(common), [f"tfe_{c}" for c in C]].sum()
    rows.append(dict(setup=k, start_local=k[:2] + ":00", level=k.split("_")[1], n=int(len(H)), frac_valid=float(TF[k].loc[TF[k].index.isin(common), "frac_valid"].mean()),
                     r2_AOD=float(mA.rsquared), r2_logPM=float(mP.rsquared), **{f"exp_{c}": float(100 * ex[f"tfe_{c}"] / ex.sum()) for c in C},
                     **{f"loss_{c}": float(100 * phi[c] / joint) for c in C}, joint_MWh=float(joint)))
    print(k, {c: round(rows[-1][f"loss_{c}"], 1) for c in C}, f"{time.time() - t0:.0f}s", flush=True)
S = pd.DataFrame(rows); S.to_csv(f"{OUT}/traj_sensitivity_shares.csv", index=False)
json.dump(dict(check_vs_original=CHK, n_common=int(len(common)), table=rows), open(f"{OUT}/traj_sensitivity.json", "w"), indent=1)
print(S.round(1).to_string(index=False)); print(f"total {time.time() - t0:.0f}s")
