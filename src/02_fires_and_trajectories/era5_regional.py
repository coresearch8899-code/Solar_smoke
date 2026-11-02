import sys, numpy as np, xarray as xr, warnings; warnings.filterwarnings("ignore")
ds = xr.open_dataset(sys.argv[1], engine="cfgrib", backend_kwargs={"indexpath": ""}).load()
print(ds)
t = ds.valid_time.values if "valid_time" in ds.coords else ds.time.values
np.savez_compressed(sys.argv[2], time=t.astype("datetime64[s]").astype("int64"), lat=ds.latitude.values, lon=ds.longitude.values,
                    lev=ds.isobaricInhPa.values, u=ds.u.values.astype("float32"), v=ds.v.values.astype("float32"), w=ds.w.values.astype("float32"))
print("saved", ds.u.shape)
