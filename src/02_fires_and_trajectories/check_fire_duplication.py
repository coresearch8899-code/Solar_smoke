"""Revision check: were the Thailand-only FIRMS downloads (8231146-8) read together with the regional ones (823168-70),
double-counting Thai detections? Reproduces the filter of fires_regional.py and compares counts with the 9,377,960 reported."""
import sys, glob, pandas as pd
F, OUT = sys.argv[1:3]
rows = []
for f in sorted(glob.glob(f"{F}/DL_FIRE_*/fire_*.csv")):
    n = 0; dup_key = []
    for d in pd.read_csv(f, usecols=lambda c: c in ["latitude", "longitude", "acq_date", "acq_time", "satellite", "confidence", "frp", "type"],
                         dtype={"satellite": str, "confidence": str}, chunksize=1_000_000):
        d = d[d.confidence.isin(["n", "h"]) & d.latitude.between(10, 28) & d.longitude.between(88, 110)]
        if "type" in d: d = d[d["type"] == 0]
        n += len(d)
    rows.append(dict(file=f.split("\\")[-1], set="TH-only" if any(k in f for k in ("823146", "823147", "823148")) else "regional", kept=n))
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/R0_fire_duplication_check.csv", index=False)
print(R.to_string()); print(R.groupby("set").kept.sum()); print("all files:", R.kept.sum(), " reported in manuscript: 9,377,960")
# exact overlap of TH-only rows inside the regional set (same lat, lon, time, satellite)
k = ["latitude", "longitude", "acq_date", "acq_time"]
th = pd.concat([pd.read_csv(f, usecols=k + ["satellite"], dtype={"satellite": str}) for f in glob.glob(f"{F}/DL_FIRE_*82314[678]/fire_archive*.csv")])
rg = pd.concat([pd.read_csv(f, usecols=k + ["satellite"], dtype={"satellite": str}) for f in glob.glob(f"{F}/DL_FIRE_*8231[67][0-9]/fire_archive*.csv")])
rg = rg[rg.latitude.between(th.latitude.min(), th.latitude.max()) & rg.longitude.between(th.longitude.min(), th.longitude.max())]
m = th.merge(rg, on=k + ["satellite"], how="left", indicator=True)
print("TH-only archive rows:", len(th), " of which also in regional archive:", int((m._merge == "both").sum()))
