"""Assign ESA WorldCover 2021 class to VIIRS fire detections (100 m modal aggregation of the 10 m map)."""
import sys, glob, numpy as np, pandas as pd, rasterio
from rasterio.enums import Resampling
D, OUT = sys.argv[1], sys.argv[2]
fr = []
for tag, pat in [("SNPP","DL_FIRE_SV-C2_823148/fire_%s_SV-C2_823148.csv"),("N20","DL_FIRE_J1V-C2_823146/fire_%s_J1V-C2_823146.csv"),("N21","DL_FIRE_J2V-C2_823147/fire_%s_J2V-C2_823147.csv")]:
    for k in ("archive","nrt"):
        f = pd.read_csv(f"{D}/"+pat%k); f["sat"]=tag; fr.append(f)
fr = pd.concat(fr, ignore_index=True)
fr = fr[fr.confidence.isin(["n","h"]) & ((fr["type"].isna()) | (fr["type"]==0))].reset_index(drop=True)
fr["lc"] = 0
tiles = glob.glob(f"{D}/ESA_WorldCover*.tif") + glob.glob(f"{D}/New folder/ESA_WorldCover*.tif")
for t in tiles:
    with rasterio.open(t) as src:
        b = src.bounds
        m = fr.latitude.between(b.bottom, b.top, inclusive="left") & fr.longitude.between(b.left, b.right, inclusive="left")
        if not m.any(): continue
        arr = src.read(1, out_shape=(src.height//10, src.width//10), resampling=Resampling.mode)
        r = ((b.top - fr.loc[m,"latitude"]) / (b.top-b.bottom) * arr.shape[0]).astype(int).clip(0, arr.shape[0]-1)
        c = ((fr.loc[m,"longitude"] - b.left) / (b.right-b.left) * arr.shape[1]).astype(int).clip(0, arr.shape[1]-1)
        fr.loc[m,"lc"] = arr[r.values, c.values]
        print(t.split("/")[-1], int(m.sum()), flush=True)
fr[["latitude","longitude","acq_date","acq_time","sat","frp","lc"]].to_csv(f"{OUT}/fires_landcover.csv.gz", index=False)
