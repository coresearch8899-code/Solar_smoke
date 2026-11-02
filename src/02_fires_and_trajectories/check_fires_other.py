"""Revision check: does the drop of 'Other' trajectory fire exposure in 2025-26 reflect real fires or a data gap?
Counts Jan-May VIIRS detections / FRP per year and satellite in regions that can only be 'Other' (no Thai/Myanmar/Lao territory)."""
import sys, glob, pandas as pd
F, OUT = sys.argv[1:3]
reg = {"India_Bangladesh (lon<92)": lambda d: d.longitude < 92.0,
       "China_Yunnan (lat>23.5, lon>101)": lambda d: (d.latitude > 23.5) & (d.longitude > 101),
       "Vietnam_Cambodia (lon>107.5)": lambda d: d.longitude > 107.5,
       "whole domain": lambda d: d.latitude > -90}
rows, span = [], []
for f in sorted(glob.glob(f"{F}/DL_FIRE_*/fire_*.csv")):
    for d in pd.read_csv(f, usecols=lambda c: c in ["latitude", "longitude", "acq_date", "satellite", "confidence", "frp", "type"],
                         dtype={"satellite": str, "confidence": str}, chunksize=1_000_000):
        if "type" not in d: d["type"] = 0
        span.append(dict(file=f.split("\\")[-1], first=d.acq_date.min(), last=d.acq_date.max(), n=len(d),
                         lat_min=d.latitude.min(), lat_max=d.latitude.max(), lon_min=d.longitude.min(), lon_max=d.longitude.max()))
        d = d[d.confidence.isin(["n", "h"]) & (d["type"] == 0) & d.latitude.between(10, 28) & d.longitude.between(88, 110)]
        dt = pd.to_datetime(d.acq_date); d = d[dt.dt.month.isin([1, 2, 3, 4, 5])]; yr = dt[d.index].dt.year
        for k, m in reg.items():
            mm = m(d)
            g = d[mm].groupby([yr[mm], d.satellite[mm]]).frp.agg(["size", "sum"]).reset_index()
            g.columns = ["year", "sat", "n", "frp"]; g["region"] = k; g["file"] = f.split("\\")[-1]; rows.append(g)
S = pd.DataFrame(span).groupby("file").agg(first=("first", "min"), last=("last", "max"), n=("n", "sum"), lat_min=("lat_min", "min"),
                                          lat_max=("lat_max", "max"), lon_min=("lon_min", "min"), lon_max=("lon_max", "max"))
R = pd.concat(rows).groupby(["region", "year", "sat"])[["n", "frp"]].sum().reset_index()
S.to_csv(f"{OUT}/R0_fire_file_coverage.csv"); R.to_csv(f"{OUT}/R0_fire_other_regions_by_year.csv", index=False)
pd.set_option("display.width", 200)
print(S.to_string()); print(R.pivot_table(index=["region", "sat"], columns="year", values="n", aggfunc="sum").to_string())
