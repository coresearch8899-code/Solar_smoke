"""Parse Zone-1 on-site meteo station exports (weekly, 5-min/15-min) into one QC'd time series."""
import sys, glob, numpy as np, pandas as pd
D, OUT = sys.argv[1], sys.argv[2]
names = {"ความชื้นแวดล้อม": "rh", "แรงดันบรรยากาศ": "p_hpa", "มุมลม": "wdir", "สเกลความเร็วลม": "wscale",
         "ความเร็วลม": "ws", "อุณหภูมิโมดูล": "t_mod", "อุณหภูมิโดยรอบ": "t_amb",
         "แสงแดดที่ส่องรายวันตามความชัน": "poa_cum_Wh", "แสงแดดที่ส่องชั่วคราวตามความชัน": "poa_W",
         "แสงแดดที่ส่องระนาบแนวนอนต่อวัน": "ghi_cum_Wh", "แสงแดดที่ส่องระนาบแนวนอนชั่วคราว": "ghi_W"}
fr = []
for f in sorted(glob.glob(f"{D}/Meteo_data/*.csv")):
    d = pd.read_csv(f, skiprows=1, encoding="utf-8-sig", na_values=["--"])
    cols = {"เวลา": "time"}
    for c in d.columns[1:]:
        key = c.split("/")[1].split("(")[0]
        cols[c] = names.get(key, key)
    fr.append(d.rename(columns=cols))
m = pd.concat(fr).drop_duplicates("time")
m["time"] = pd.to_datetime(m.time); m = m.set_index("time").sort_index()
m = m.apply(pd.to_numeric, errors="coerce")
dt = m.index.to_series().diff().dt.total_seconds().div(60).mode().iloc[0]
# QC: physical limits
for c, lo, hi in (("ghi_W", -5, 1500), ("poa_W", -5, 1600), ("t_amb", 0, 50), ("t_mod", -5, 90), ("rh", 1, 100.5)):
    m.loc[~m[c].between(lo, hi), c] = np.nan
m.to_csv(f"{OUT}/meteo_zone1_clean.csv.gz")
day = m.groupby(m.index.normalize())
D_ = pd.DataFrame({"ghi_kWh": day.ghi_cum_Wh.max() / 1000, "poa_kWh": day.poa_cum_Wh.max() / 1000,
                   "ghi_int_kWh": day.ghi_W.sum() * dt / 60 / 1000, "poa_int_kWh": day.poa_W.sum() * dt / 60 / 1000,
                   "n_valid_ghi": day.ghi_W.count(), "t_amb": day.t_amb.mean(), "t_mod_max": day.t_mod.max(),
                   "rh": day.rh.mean(), "ws": day.ws.mean()})
D_.to_csv(f"{OUT}/meteo_zone1_daily.csv")
print("interval_min", dt, "rows", len(m), m.index.min(), m.index.max())
print(m.notna().mean().round(3).to_string())
print(D_.describe().round(2).to_string())
