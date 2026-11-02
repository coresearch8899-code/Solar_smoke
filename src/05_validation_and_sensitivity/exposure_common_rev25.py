"""Exposure shares on the same days as the attributed losses (Table 4): January-May haze-model days with complete covariates and lagged
exposure, from commissioning (3 March 2023), compared with the shares over all January-May trajectory days (which include January-February
2023). Usage: exposure_common_rev25.py <step3_csv> <step3_dir_with_tfe> <out_json>"""
import sys, json, numpy as np, pandas as pd
D3, TD, OUT = sys.argv[1:4]
C = ["THA", "MMR", "LAO", "OTH"]; MET = ["lblh", "inv850_07", "e5_rh", "e5_ws", "e5_tp", "ws850", "e5_tcwv"]
d = pd.read_csv(D3, index_col=0, parse_dates=True); R = {}
for k in ("p850", "p700", "3d850"):
    t = pd.read_csv(f"{TD}/tfe_{k}.csv", index_col=0, parse_dates=True); tj = t[t.index.month.isin([1, 2, 3, 4, 5])]
    H = d.drop(columns=[f"tfe_{c}" for c in C], errors="ignore").join(t, how="inner"); H = H[H.index.month.isin([1, 2, 3, 4, 5])].copy()
    H["lblh"] = np.log(H.e5_blh_day)
    for c in C:
        for L in (0, 1): H[f"lt_{c}_{L}"] = np.log1p(H[f"tfe_{c}"].shift(L))
    H = H.dropna(subset=[f"lt_{c}_{L}" for c in C for L in (0, 1)] + MET + ["pm25", "aod_ceres"]); H = H[H.index >= "2023-03-03"]
    sh = lambda df: {c: float(100 * df[f"tfe_{c}"].sum() / df[[f"tfe_{x}" for x in C]].sum().sum()) for c in C}
    R[k] = dict(all_days=sh(tj), n_all=int(len(tj)), loss_days=sh(H), n_loss=int(len(H)),
                annual_MMR_loss_days={int(y): round(sh(g)["MMR"], 1) for y, g in H.groupby(H.index.year)})
json.dump(R, open(OUT, "w"), indent=1); print(json.dumps(R, indent=1))
