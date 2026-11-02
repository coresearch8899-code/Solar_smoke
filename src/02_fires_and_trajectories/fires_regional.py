"""Regional VIIRS fires (SNPP, NOAA-20, NOAA-21) -> filtered, country-tagged, local-date table."""
import sys, glob, numpy as np, pandas as pd, geopandas as gpd
F, G, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
fr = []
for f in glob.glob(f"{F}/*/fire_*.csv"):
    for d in pd.read_csv(f, usecols=lambda c: c in ["latitude", "longitude", "acq_date", "acq_time", "satellite", "confidence", "frp", "type"],
                         dtype={"satellite": str, "confidence": str}, chunksize=1_000_000):
        d = d[d.confidence.isin(["n", "h"]) & d.latitude.between(10, 28) & d.longitude.between(88, 110)]
        if "type" in d: d = d[d["type"] == 0].drop(columns="type")
        fr.append(d.astype({"latitude": "float32", "longitude": "float32", "frp": "float32"}))
fr = pd.concat(fr, ignore_index=True)
n0 = len(fr)
t = pd.to_datetime(fr.acq_date) + pd.to_timedelta(fr.acq_time // 100, "h") + pd.to_timedelta(fr.acq_time % 100, "m")
fr["t_utc"] = t; fr["date"] = (t + pd.Timedelta(hours=7)).dt.normalize()
fr["sat"] = fr.satellite.astype(str).replace({"N": "SNPP", "1": "N20", "2": "N21", "Suomi NPP": "SNPP", "NOAA-20": "N20", "NOAA-21": "N21"})
g = gpd.GeoDataFrame(fr, geometry=gpd.points_from_xy(fr.longitude, fr.latitude), crs=4326)
adm = pd.concat([gpd.read_file(f"{G}/gadm41_{c}.gpkg", layer="ADM_ADM_1")[["GID_0", "NAME_1", "geometry"]] for c in ("THA", "MMR", "LAO")])
g = gpd.sjoin(g, adm, how="left", predicate="within").drop(columns=["index_right", "geometry"])
g["country"] = g.GID_0.fillna("OTH")
lat0, lon0 = np.radians(19.905), np.radians(99.83); la, lo = np.radians(g.latitude), np.radians(g.longitude)
g["dist_km"] = 2 * 6371 * np.arcsin(np.sqrt(np.sin((la - lat0) / 2) ** 2 + np.cos(lat0) * np.cos(la) * np.sin((lo - lon0) / 2) ** 2))
g = g.drop(columns=["confidence", "satellite", "acq_time", "acq_date", "GID_0"])
g.to_parquet(f"{OUT}/fires_regional.parquet")
print("raw", n0, "kept", len(g)); print(g.sat.value_counts()); print(g.country.value_counts())
print(g[g.date.dt.month.isin([1,2,3,4,5])].groupby([g.date.dt.year, "country"]).frp.sum().unstack().round(0))
