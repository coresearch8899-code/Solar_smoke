"""Extract ERA5 (CDS GRIB) at the CRH site (19.905N, 99.83E) by bilinear interpolation -> hourly CSV (local time, UTC+7)."""
import sys, glob, warnings, numpy as np, pandas as pd, xarray as xr
warnings.filterwarnings("ignore")
D, OUT = sys.argv[1], sys.argv[2]
LAT, LON = 19.905, 99.83
def to_frame(da):
    da = da.interp(latitude=LAT, longitude=LON)
    if "step" in da.dims:                      # accumulated fields: (time, step) -> valid_time
        vt = da.valid_time.values.ravel(); v = da.values.ravel()
    else:
        vt = da.valid_time.values if "valid_time" in da.coords else da.time.values; v = da.values
    s = pd.Series(v, index=pd.to_datetime(vt)).dropna()
    return s[~s.index.duplicated()].sort_index()
sl, pl = {}, {}
for f in sorted(glob.glob(f"{D}/*.grib") + glob.glob(f"{D}/Era5/*.grib")):
    ds = xr.open_dataset(f, engine="cfgrib", backend_kwargs={"indexpath": ""}).load()
    for v in ds.data_vars:
        da = ds[v]
        if "isobaricInhPa" in da.dims:
            for lev in (1000, 975, 950, 925, 900, 850, 800, 700, 600, 500):
                if lev in da.isobaricInhPa.values:
                    s = to_frame(da.sel(isobaricInhPa=lev)); k = f"{v}{lev}"
                    pl[k] = pd.concat([pl[k], s]) if k in pl else s
        else:
            s = to_frame(da); sl[v] = pd.concat([sl[v], s]) if v in sl else s
    print(f.split("/")[-1], list(ds.data_vars), flush=True)
S = pd.DataFrame({k: v[~v.index.duplicated()] for k, v in sl.items()}).sort_index()
P = pd.DataFrame({k: v[~v.index.duplicated()].sort_index() for k, v in pl.items()}).sort_index()
for X, name in ((S, "era5_sl_site_hourly.csv"), (P, "era5_pl_site_3hourly.csv")):
    X.index = X.index + pd.Timedelta(hours=7); X.index.name = "time_local"
    X.to_csv(f"{OUT}/{name}")
    print(name, X.shape, X.index.min(), X.index.max())
