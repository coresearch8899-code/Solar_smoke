"""Revision check: Jan-May VIIRS detections/FRP per year in the NE-India / Chittagong strip west of Myanmar (92-93.2E, 21.5-25.5N),
which 850 hPa westerly trajectories cross just before Myanmar and which GADM tags as 'Other'."""
import sys, glob, pandas as pd
F, OUT = sys.argv[1:3]
rows = []
for f in sorted(glob.glob(f"{F}/DL_FIRE_*8231[67][0-9]/fire_*.csv")):
    for d in pd.read_csv(f, usecols=lambda c: c in ["latitude", "longitude", "acq_date", "satellite", "confidence", "frp", "type"],
                         dtype={"satellite": str, "confidence": str}, chunksize=1_000_000):
        if "type" not in d: d["type"] = 0
        d = d[d.confidence.isin(["n", "h"]) & (d["type"] == 0) & d.latitude.between(21.5, 25.5) & d.longitude.between(92.0, 93.2)]
        dt = pd.to_datetime(d.acq_date); d = d[dt.dt.month.isin([1, 2, 3, 4, 5])]
        rows.append(d.assign(year=dt[d.index].dt.year)[["year", "satellite", "frp"]])
R = pd.concat(rows).groupby(["year", "satellite"]).frp.agg(["size", "sum"]).round(0)
R.to_csv(f"{OUT}/R0_fire_NE_India_strip.csv"); print(R.unstack().to_string())
