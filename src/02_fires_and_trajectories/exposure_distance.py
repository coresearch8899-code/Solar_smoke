"""Distance from the hospital at which each country's trajectory fire exposure was picked up (850 hPa, Jan-May 2023-2026).
Re-uses the exposure rule of trajectories.py (0.5 deg FRP grid on the local date, satellite-normalised) and weights the great-circle
distance of every hourly trajectory point by the exposure it collected. Checks the distances once quoted in Section 4.1."""
import sys, json, numpy as np, pandas as pd
TR, FIRES, OUT = sys.argv[1:4]
LAT0, LON0 = 19.905, 99.83
tr = np.load(f"{TR}/traj_p850.npy"); tfe = pd.read_csv(f"{TR}/tfe_p850.csv", index_col=0, parse_dates=True)
jm = tfe.index.month.isin([1, 2, 3, 4, 5]); tr = tr[jm]
f = pd.read_parquet(FIRES, columns=["latitude", "longitude", "date", "sat", "frp", "country"])
f = f[f.date.dt.month.isin([12, 1, 2, 3, 4, 5])]
nsat = f.groupby(["date", "sat"]).size().unstack().notna().sum(axis=1); f["frp_w"] = f.frp / f.date.map(nsat)
G = 0.5; glat = np.arange(10, 28, G); glon = np.arange(88, 110, G)
f["iy"] = ((f.latitude - 10) // G).astype(int).clip(0, len(glat) - 1); f["ix"] = ((f.longitude - 88) // G).astype(int).clip(0, len(glon) - 1)
days = pd.date_range(f.date.min(), f.date.max(), freq="D"); dindex = {d: i for i, d in enumerate(days)}
C = ["THA", "MMR", "LAO", "OTH"]; grid = np.zeros((len(C), len(days), len(glat), len(glon)), "float32")
for ci, c in enumerate(C):
    s = f[f.country == c].groupby(["date", "iy", "ix"]).frp_w.sum().reset_index()
    grid[ci, s.date.map(dindex).values, s.iy.values, s.ix.values] = s.frp_w.values
tt = pd.to_datetime((tr[:, :, 0] * 3600).astype("int64"), unit="s").values.reshape(tr.shape[0], -1)
di = pd.Series((pd.DatetimeIndex(tt.ravel()) + pd.Timedelta(hours=7)).normalize()).map(dindex).values.reshape(tr.shape[0], -1)
iy = (tr[:, :, 1] - 10) // G; ix = (tr[:, :, 2] - 88) // G
valid = ~np.isnan(tr[:, :, 1]) & (iy >= 0) & (iy < len(glat)) & (ix >= 0) & (ix < len(glon)) & ~pd.isna(di)
la, lo = np.radians(tr[:, :, 1]), np.radians(tr[:, :, 2]); la0, lo0 = np.radians(LAT0), np.radians(LON0)
dist = 2 * 6371 * np.arcsin(np.sqrt(np.sin((la - la0) / 2) ** 2 + np.cos(la0) * np.cos(la) * np.sin((lo - lo0) / 2) ** 2))
R = {}
for ci, c in enumerate(C):
    e = np.zeros(tr.shape[:2]); e[valid] = grid[ci, di[valid].astype(int), iy[valid].astype(int), ix[valid].astype(int)]
    e, dd = e[:, 1:].ravel(), dist[:, 1:].ravel(); m = e > 0; w, x = e[m], dd[m]; o = np.argsort(x); cw = np.cumsum(w[o]) / w.sum()
    q = lambda p: float(x[o][np.searchsorted(cw, p)])
    R[c] = dict(median_km=round(q(0.5)), p10_km=round(q(0.1)), p90_km=round(q(0.9)), share_within_300km=round(100 * w[x <= 300].sum() / w.sum(), 1),
                share_300_700km=round(100 * w[(x > 300) & (x <= 700)].sum() / w.sum(), 1))
json.dump(R, open(f"{OUT}/exposure_distance_p850.json", "w"), indent=1); print(json.dumps(R, indent=1))
