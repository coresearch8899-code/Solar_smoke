"""72-h kinematic back-trajectories from ERA5 (0.25 deg, 6-hourly u/v/w at 850/700/500 hPa), started daily at
06 UTC (13:00 local) over the CRH site, plus trajectory-weighted fire exposure (TFE) by country.
Two set-ups: isobaric 850 hPa and 700 hPa (main), and 3-D from 850 hPa using omega (sensitivity)."""
import sys, numpy as np, pandas as pd
Z, FIRES, OUT = sys.argv[1:4]
LAT0, LON0 = 19.905, 99.83
z = np.load(Z)
T = z["time"].astype("datetime64[s]").astype("int64") / 3600.0           # hours
LAT, LON, LEV = z["lat"], z["lon"], z["lev"]                           # lat descending
U, V, W = z["u"], z["v"], z["w"]
R = 6371e3

def interp(F, t, p, la, lo):
    """F[time, lev, lat, lon]; t hours, p hPa (scalar arrays), linear in t/p, bilinear in space."""
    it = np.searchsorted(T, t) - 1
    ok = (it >= 0) & (it < len(T) - 1) & ((T[np.clip(it + 1, 0, len(T) - 1)] - T[np.clip(it, 0, len(T) - 1)]) <= 6.01)
    it = np.clip(it, 0, len(T) - 2); ft = (t - T[it]) / (T[it + 1] - T[it])
    # pressure: LEV = [850, 700, 500] (decreasing)
    pc = np.clip(np.nan_to_num(p, nan=850.0), 500, 850)
    ip = np.where(pc >= 700, 0, 1); fp = (LEV[ip] - pc) / (LEV[ip] - LEV[ip + 1])
    y = (LAT[0] - la) / 0.25; x = (lo - LON[0]) / 0.25
    inside = (y >= 0) & (y <= len(LAT) - 1.001) & (x >= 0) & (x <= len(LON) - 1.001)
    inside = inside & np.isfinite(y) & np.isfinite(x)
    y = np.clip(np.nan_to_num(y), 0, len(LAT) - 1.001); x = np.clip(np.nan_to_num(x), 0, len(LON) - 1.001)
    iy, ix = y.astype(int), x.astype(int); fy, fx = y - iy, x - ix
    out = 0
    for dt_, wt in ((0, 1 - ft), (1, ft)):
        for dp, wp in ((0, 1 - fp), (1, fp)):
            a, b = it + dt_, ip + dp
            v = (F[a, b, iy, ix] * (1 - fy) * (1 - fx) + F[a, b, iy + 1, ix] * fy * (1 - fx) +
                 F[a, b, iy, ix + 1] * (1 - fy) * fx + F[a, b, iy + 1, ix + 1] * fy * fx)
            out = out + wt * wp * v
    return np.where(ok & inside, out, np.nan)

def run(starts, p0, vertical):
    n = len(starts); la = np.full(n, LAT0); lo = np.full(n, LON0); p = np.full(n, float(p0)); t = starts.astype(float)
    traj = [np.stack([t, la, lo, p], 1)]
    for step in range(72):                                  # 1-h steps backwards, Heun scheme
        dt = -3600.0
        def vel(t_, la_, lo_, p_):
            u = interp(U, t_, p_, la_, lo_); v = interp(V, t_, p_, la_, lo_)
            w = interp(W, t_, p_, la_, lo_) if vertical else np.zeros_like(u)
            return u, v, w
        u1, v1, w1 = vel(t, la, lo, p)
        la1 = la + np.degrees(v1 * dt / R); lo1 = lo + np.degrees(u1 * dt / (R * np.cos(np.radians(la)))); p1 = p + w1 * dt / 100
        u2, v2, w2 = vel(t - 1, la1, lo1, p1)
        la = la + np.degrees(0.5 * (v1 + v2) * dt / R)
        lo = lo + np.degrees(0.5 * (u1 + u2) * dt / (R * np.cos(np.radians(la))))
        p = np.clip(p + 0.5 * (w1 + w2) * dt / 100, 500, 850) if vertical else p
        t = t - 1
        traj.append(np.stack([t, la, lo, p], 1))
    return np.stack(traj, 1)                                # [n, 73, 4]

# start times: every day 06 UTC with full 72 h of wind data behind it
start_dt = pd.to_datetime(z["time"].astype("datetime64[s]"))
starts = start_dt[(start_dt.hour == 6)]
first = pd.Series(start_dt).groupby(start_dt.year).min()
starts = starts[[(s - pd.Timedelta(hours=72)) >= first[s.year] for s in starts]]
st_h = (starts.values.astype("datetime64[s]").astype("int64") / 3600.0)

# fire grid (0.5 deg, daily local date, FRP normalised by satellites active that day)
f = pd.read_parquet(FIRES, columns=["latitude", "longitude", "date", "sat", "frp", "country"])
f = f[f.date.dt.month.isin([12, 1, 2, 3, 4, 5])]
nsat = f.groupby(["date", "sat"]).size().unstack().notna().sum(axis=1)
f["frp_w"] = f.frp / f.date.map(nsat)
G = 0.5; glat = np.arange(10, 28, G); glon = np.arange(88, 110, G)
f["iy"] = ((f.latitude - 10) // G).astype(int).clip(0, len(glat) - 1); f["ix"] = ((f.longitude - 88) // G).astype(int).clip(0, len(glon) - 1)
days = pd.date_range(f.date.min(), f.date.max(), freq="D"); dindex = {d: i for i, d in enumerate(days)}
countries = ["THA", "MMR", "LAO", "OTH"]
grid = np.zeros((len(countries), len(days), len(glat), len(glon)), "float32")
for ci, c in enumerate(countries):
    s = f[f.country == c].groupby(["date", "iy", "ix"]).frp_w.sum().reset_index()
    grid[ci, s.date.map(dindex).values, s.iy.values, s.ix.values] = s.frp_w.values

res = {}
for name, p0, vert in (("p850", 850, False), ("p700", 700, False), ("3d850", 850, True)):
    tr = run(st_h, p0, vert)
    np.save(f"{OUT}/traj_{name}.npy", tr.astype("float32"))
    # exposure: each hourly point picks up FRP in its 0.5-deg cell on its local date (point counted once per hour)
    tt = pd.to_datetime((tr[:, :, 0] * 3600).astype("int64"), unit="s").values.reshape(tr.shape[0], -1)
    loc_date = (pd.DatetimeIndex(tt.ravel()) + pd.Timedelta(hours=7)).normalize()
    di = pd.Series(loc_date).map(dindex).values.reshape(tr.shape[0], -1)
    iy = ((tr[:, :, 1] - 10) // G); ix = ((tr[:, :, 2] - 88) // G)
    valid = ~np.isnan(tr[:, :, 1]) & (iy >= 0) & (iy < len(glat)) & (ix >= 0) & (ix < len(glon)) & ~pd.isna(di)
    out = pd.DataFrame(index=starts.normalize())
    for ci, c in enumerate(countries):
        e = np.zeros(tr.shape[:2])
        e[valid] = grid[ci, di[valid].astype(int), iy[valid].astype(int), ix[valid].astype(int)]
        out[f"tfe_{c}"] = e[:, 1:].sum(1)                                  # MW-h of upwind fire exposure
        out[f"tfe24_{c}"] = e[:, 1:25].sum(1)
    out["frac_valid"] = valid[:, 1:].mean(1)
    out["end_lat"] = tr[:, -1, 1]; out["end_lon"] = tr[:, -1, 2]
    out.to_csv(f"{OUT}/tfe_{name}.csv")
    res[name] = out
    tot = out[[f"tfe_{c}" for c in countries]].sum()
    print(name, "n", len(out), "valid", round(out.frac_valid.mean(), 3), "share %", (100 * tot / tot.sum()).round(1).to_dict(), flush=True)
r = pd.concat({k: v[[f"tfe_{c}" for c in countries]].sum(1) for k, v in res.items()}, axis=1)
print("corr total TFE between set-ups\n", np.log1p(r).corr().round(3))
