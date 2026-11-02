"""Step 6a – 72-h back-trajectories from HOURLY ERA5 winds (N26 S14 W92 E106, 2021-2025) and trajectory fire exposure (TFE).
Set-ups (all started daily 06 UTC from the site, Heun scheme with 1-h steps, as in trajectories.py):
  h850     isobaric 850 hPa, hourly winds
  h850_6h  the same winds sub-sampled to 00/06/12/18 UTC (isolates the effect of the 6-hourly winds used originally)
  blproxy  mean of the 10-m and 850-hPa winds, a proxy for transport inside the boundary layer
Fires: NOAA-20 (J1) VIIRS; country from the FIRMS country files (Thailand, Myanmar, Lao PDR); 'other' = regional J1
detections (2023-) not present in the country files. 2021-2022 have no 'other' class (regional file starts 2023)."""
import sys, glob, h5py, numpy as np, pandas as pd
ERA, FIRMS_C, FIRMS_R, OUT = sys.argv[1:5]
LAT0, LON0 = 19.905, 99.83; G = 0.5; R = 6371e3
C = ["THA", "MMR", "LAO", "OTH"]; CMAP = {"Thailand": "THA", "Myanmar": "MMR", "Lao_PDR": "LAO", "Lao PDR": "LAO", "Laos": "LAO"}

# ---------------- fires (J1, Dec-May, n/h confidence, vegetation type) ----------------
def clean(df):
    df = df[df.confidence.astype(str).isin(["n", "h"])]
    if "type" in df: df = df[df["type"] == 0]
    df = df.assign(acq_time=df.acq_time.astype(int), latitude=df.latitude.round(5), longitude=df.longitude.round(5))
    return df[df.latitude.between(14, 26) & df.longitude.between(92, 106) & pd.to_datetime(df.acq_date).dt.month.isin([12, 1, 2, 3, 4, 5])]
fc = clean(pd.read_csv(FIRMS_C, usecols=["country_file", "latitude", "longitude", "acq_date", "acq_time", "confidence", "frp", "type"], dtype={"confidence": str}))
fc["country"] = fc.country_file.map(CMAP)
assert fc.country.notna().all(), fc.country_file.unique()
fr = []
for f in glob.glob(f"{FIRMS_R}/fire_*.csv"):
    d = pd.read_csv(f, usecols=lambda c: c in ["latitude", "longitude", "acq_date", "acq_time", "confidence", "frp", "type"], dtype={"confidence": str})
    fr.append(clean(d))
fr = pd.concat(fr); fr = fr[pd.to_datetime(fr.acq_date) <= "2025-12-31"]
key = ["latitude", "longitude", "acq_date", "acq_time"]
m = fr.merge(fc[key + ["country"]].drop_duplicates(key), on=key, how="left")
# match check inside an all-Thai box (15-18N, 99.5-101.5E): nearly every regional detection should carry a country
inner = m[m.latitude.between(15, 18) & m.longitude.between(99.5, 101.5)]
print("regional J1 rows", len(m), "matched", round(100 * m.country.notna().mean(), 1), "% | all-Thai box matched", round(100 * inner.country.notna().mean(), 2), "%")
oth = m[m.country.isna()].assign(country="OTH")
fires = pd.concat([fc[key + ["frp", "country"]], oth[key + ["frp", "country"]]], ignore_index=True)
t = pd.to_datetime(fires.acq_date) + pd.to_timedelta(fires.acq_time // 100, "h") + pd.to_timedelta(fires.acq_time % 100, "m")
fires["date"] = (t + pd.Timedelta(hours=7)).dt.normalize()
fires["iy"] = ((fires.latitude - 14) // G).astype(int).clip(0, 23); fires["ix"] = ((fires.longitude - 92) // G).astype(int).clip(0, 27)
fires.groupby([fires.date.dt.year, "country"]).frp.agg(["size", "sum"]).round(0).to_csv(f"{OUT}/fires_J1_by_year_country.csv")
days = pd.date_range("2020-12-01", "2025-12-31", freq="D"); dindex = {d_: i for i, d_ in enumerate(days)}
grid = np.zeros((4, len(days), 24, 28), "float32")
for ci, c in enumerate(C):
    s = fires[fires.country == c].groupby(["date", "iy", "ix"]).frp.sum().reset_index()
    s = s[s.date.isin(dindex)]                                                  # drops local dates past 31 Dec 2025
    grid[ci, s.date.map(dindex).values, s.iy.values, s.ix.values] = s.frp.values
has_oth = pd.Series(days >= pd.Timestamp("2023-01-01"), index=days)

# ---------------- winds ----------------
def load(var, months):
    arrs, times = [], []
    for ym in months:
        with h5py.File(f"{ERA}/{var}/era5_{var}_N26W92S14E106_{ym}.nc", "r") as h:
            arrs.append(h[var][:]); times.append(h["time"][:])
            lat, lon = h["latitude"][:], h["longitude"][:]
    return np.concatenate(arrs), np.concatenate(times) / 3600.0, lat, lon

def interp(F, T, t, la, lo, lat, lon):
    it = np.clip(np.searchsorted(T, t) - 1, 0, len(T) - 2); ft = (t - T[it]) / (T[it + 1] - T[it])
    y = (lat[0] - la) / 0.25; x = (lo - lon[0]) / 0.25
    inside = np.isfinite(y) & np.isfinite(x) & (y >= 0) & (y <= len(lat) - 1.001) & (x >= 0) & (x <= len(lon) - 1.001)
    y = np.clip(np.nan_to_num(y), 0, len(lat) - 1.001); x = np.clip(np.nan_to_num(x), 0, len(lon) - 1.001)
    iy, ix = y.astype(int), x.astype(int); fy, fx = y - iy, x - ix
    out = 0
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
    out["frac_valid"] = valid[:, 1:].mean(1); out["end_lat"] = tr[:, -1, 1]; out["end_lon"] = tr[:, -1, 2]
    for c in ("OTH",):
        out.loc[~has_oth.reindex(out.index).values, [f"tfe_{c}", f"tfe24_{c}"]] = np.nan
    return out

res = {k: [] for k in ("h850", "h850_6h", "blproxy")}
for yr in range(2021, 2026):
    months = ([f"{yr - 1}12"] if yr > 2021 else []) + [f"{yr}{m:02d}" for m in range(1, 6)]
    U8, T, lat, lon = load("u850", months); V8 = load("v850", months)[0]; U10 = load("10u", months)[0]; V10 = load("10v", months)[0]
    st = pd.date_range(f"{yr}-01-01 06:00", f"{yr}-05-31 06:00", freq="D")
    st_h = (st.values.astype("datetime64[s]").astype("int64") / 3600.0); st_h = st_h[st_h - 72 >= T[0]]; sd = pd.to_datetime(st_h * 3600, unit="s").normalize()
    sub = (np.round(T) % 6 == 0)
    for name, (U, V, TT) in {"h850": (U8, V8, T), "h850_6h": (U8[sub], V8[sub], T[sub]), "blproxy": ((U8 + U10) / 2, (V8 + V10) / 2, T)}.items():
        tr = run(U, V, TT, lat, lon, st_h); res[name].append(exposure(tr, sd))
        if name == "h850": np.save(f"{OUT}/traj_h850_{yr}.npy", tr.astype("float32"))
    print(yr, "done", len(sd), flush=True)
for k, v in res.items():
    o = pd.concat(v); o.to_csv(f"{OUT}/tfe_{k}.csv")
    s = o[o.index.year >= 2023][[f"tfe_{c}" for c in C]].sum(); print(k, "2023-25 share %", (100 * s / s.sum()).round(1).to_dict(), "valid", round(o.frac_valid.mean(), 3))
