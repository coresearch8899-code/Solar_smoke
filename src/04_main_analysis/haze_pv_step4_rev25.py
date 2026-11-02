"""rev25 (partial-cleaning dose). Step 4 – uncertainty (moving-block bootstrap), trajectory-set-up robustness of country attribution,
economic/carbon valuation, PM2.5-dose-triggered cleaning optimisation and fire-reduction scenarios. Python only."""
import sys, json, warnings, numpy as np, pandas as pd, statsmodels.api as sm
warnings.filterwarnings("ignore")
D3, TFE_DIR, OUT = sys.argv[1:4]
rng = np.random.default_rng(42)
KWP = {"Z1": 2256.24, "Z2": 207.06, "Z3": 49.90}; KWP_TOT = sum(KWP.values())
CLEAN = pd.to_datetime(["2023-05-06", "2024-03-09", "2024-06-23", "2025-04-25", "2025-12-26", "2026-03-23", "2026-09-05"])
TARIFF, EF, CLEAN_COST = 4.2, 0.4857, 100_000.0          # THB/kWh, tCO2/MWh (TGO 2021), THB per fleet wash (assumption)
C = ["THA", "MMR", "LAO", "OTH"]; R = {}
d = pd.read_csv(D3, index_col=0, parse_dates=True)
AOD_BG = d.loc[d.index.month.isin([11, 12, 1]), "aod_ceres"].median()

# ---------------- PV model (as in step 3, dose metric, 2 mm rain reset) ----------------
COMMISSION = pd.Timestamp("2023-03-03")   # step-9 fix: new modules start clean (dose reset), as in step 3 / Table 3
PF = 2.0   # rev25: rain depth (mm) that cleans completely; a day with P mm removes the fraction min(1, P/PF) of the dose
def soil_dose(pm, tp, thr=PF, clean=CLEAN):
    """rev25 partial-cleaning dose (mg m-3 d): D_t = D_(t-1) (1 - min(1, P_t/thr)) + PM2.5_t/1000; washes and commissioning reset."""
    wash = pm.index.isin(clean + pd.Timedelta(days=1)) | (pm.index == COMMISSION)
    keep = 1 - np.minimum(1.0, tp.reindex(pm.index).fillna(0).values / thr); x = pm.values / 1000.0
    out = np.full(len(pm), np.nan); D = 0.0
    for i in range(len(pm)):
        D = 0.0 if wash[i] else D * keep[i]
        if not np.isnan(x[i]): D += x[i]; out[i] = D
    return pd.Series(out, index=pm.index)
d["dose"] = soil_dose(d.pm25, d.e5_tp)
rows = []
for z in KWP:
    t = pd.DataFrame({"ratio": d[f"Y_{z}"] / d.e5_ssrd, "aod": d.aod_ceres, "t2m": d.e5_t2m, "soil": d.dose,
                      "clear": (d.e5_tcc_day < 30) & (d.ceres_all / d.ceres_clr > 0.85) & (d.e5_tp < 1),
                      "yr": (d.index - pd.Timestamp("2023-03-03")).days / 365.25, "zone": z}, index=d.index)
    med = t.ratio.median(); t = t[t.clear & t.ratio.between(0.4 * med, 1.3 * med)]
    rows.append(t)
PV = pd.concat(rows).dropna()
def pv_fit(df):
    X = pd.get_dummies(df.zone).astype(float)
    for z in KWP: X[f"tr_{z}"] = (df.zone == z) * df.yr
    X[["aod", "t2m", "soil"]] = df[["aod", "t2m", "soil"]]
    return sm.OLS(df.ratio, X).fit()
m0 = pv_fit(PV); B_AOD, B_SOIL = m0.params.aod, m0.params.soil

# ---------------- haze models (Jan-May) with trajectory exposure ----------------
def haze_frame(tfe):
    H = d.join(tfe[[f"tfe_{c}" for c in C]], rsuffix="_x", how="inner")
    for c in C: H[f"tfe_{c}"] = H[f"tfe_{c}_x"] if f"tfe_{c}_x" in H else H[f"tfe_{c}"]
    H = H[H.index.month.isin([1, 2, 3, 4, 5])].copy()
    H["lblh"] = np.log(H.e5_blh_day); doy = H.index.dayofyear
    H["s1"], H["c1"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    for c in C:
        for L in (0, 1): H[f"lt_{c}_{L}"] = np.log1p(H[f"tfe_{c}"].shift(L))
    return H
MET = ["lblh", "inv850_07", "e5_rh", "e5_ws", "e5_tp", "ws850", "e5_tcwv", "s1", "c1"]
FIRE = [f"lt_{c}_{L}" for c in C for L in (0, 1)]
def attribution(H, b_aod, b_soil, scale=None):
    """Return fire-attributable PV loss (MWh, Jan-May, summed over years) by country; P10 counterfactual, bounded."""
    Hm = H.dropna(subset=FIRE + MET + ["pm25", "aod_ceres"])
    X = sm.add_constant(Hm[FIRE + MET])
    mA = sm.OLS(Hm.aod_ceres, X).fit(); mP = sm.OLS(np.log(Hm.pm25), X).fit()
    out = {}
    for c in C + ["ALL"]:
        cols = [x for x in FIRE if c == "ALL" or f"_{c}_" in x]
        Z = Hm[FIRE + MET].copy()
        for cc in cols:
            Z[cc] = np.minimum(Z[cc], Hm[cc].quantile(0.10)) if scale is None else np.log1p(np.expm1(Z[cc]) * scale)
        Zc = sm.add_constant(Z, has_constant="add")
        dA = np.minimum((mA.predict(X) - mA.predict(Zc)).clip(lower=0), (Hm.aod_ceres - AOD_BG).clip(lower=0))
        dP = Hm.pm25 * (1 - np.exp(-(mP.predict(X) - mP.predict(Zc)).clip(lower=0)))
        e = d.reindex(Hm.index)
        atm = (-b_aod * dA * e.e5_ssrd * (e.ceres_all / e.ceres_clr).clip(0, 1) * KWP_TOT * (Hm.index >= COMMISSION)).sum() / 1000   # rev25
        pm_c = d.pm25.copy(); pm_c.loc[Hm.index] = pm_c.loc[Hm.index] - dP
        dep_s = -b_soil * (d.dose - soil_dose(pm_c, d.e5_tp)) * d.e5_ssrd * KWP_TOT
        dep = dep_s[dep_s.index.month.isin([1, 2, 3, 4, 5, 6]) & (dep_s.index >= COMMISSION)].sum() / 1000
        out[c] = (atm, dep)
    return out, mA, mP
TFE = {k: pd.read_csv(f"{TFE_DIR}/tfe_{k}.csv", index_col=0, parse_dates=True) for k in ("p850", "p700", "3d850")}
base_att, mA0, mP0 = attribution(haze_frame(TFE["p850"]), B_AOD, B_SOIL)
def shares(att):
    tot = {c: sum(att[c]) for c in C}; s = sum(tot.values()); return {c: 100 * tot[c] / s for c in C}
R["attribution_MWh_4yr"] = {c: dict(atm=round(v[0], 1), dep=round(v[1], 1)) for c, v in base_att.items()}
R["share_p850"] = {k: round(v, 1) for k, v in shares(base_att).items()}
for k in ("p700", "3d850"):
    a, _, _ = attribution(haze_frame(TFE[k]), B_AOD, B_SOIL); R[f"share_{k}"] = {c: round(v, 1) for c, v in shares(a).items()}

# ---------------- loss decomposition (all days) ----------------
E = pd.DataFrame(index=d.index)
E["atm"] = -B_AOD * (d.aod_ceres - AOD_BG).clip(lower=0) * d.e5_ssrd * (d.ceres_all / d.ceres_clr).clip(0, 1) * KWP_TOT
E["dep"] = -B_SOIL * d.dose * d.e5_ssrd * KWP_TOT
E["act"] = sum(d[f"Y_{z}"] * k for z, k in KWP.items())
E = E[E.act.notna()]
yr = E.groupby(E.index.year).sum() / 1000
yr["loss_pct"] = 100 * (yr.atm + yr.dep) / (yr.act + yr.atm + yr.dep)
full = yr.loc[[2024, 2025]]                                                  # complete calendar years
R["annual_full_years"] = dict(atm_MWh=float(full.atm.mean()), dep_MWh=float(full.dep.mean()), loss_pct=float(full.loss_pct.mean()),
                              THB=float((full.atm + full.dep).mean() * 1000 * TARIFF), tCO2=float((full.atm + full.dep).mean() * EF))

# ---------------- moving-block bootstrap (14-day blocks, B = 400) ----------------
weeks = (PV.index - PV.index.min()).days // 14
blocks = np.unique(weeks); Hb = haze_frame(TFE["p850"]); hblocks = (Hb.index - Hb.index.min()).days // 14
boot = []
for b in range(400):
    pick = rng.choice(blocks, len(blocks), replace=True)
    PVb = pd.concat([PV[weeks == k] for k in pick])
    mb = pv_fit(PVb)
    boot.append(dict(b_aod=mb.params.aod, b_soil=mb.params.soil,
                     pct_aod=100 * mb.params.aod / PVb.ratio.mean(), pct_soil=100 * mb.params.soil / PVb.ratio.mean()))
BT = pd.DataFrame(boot); BT.to_csv(f"{OUT}/pv_bootstrap_replicates.csv", index=False)   # step-9: keep replicates (unrounded CI)
R["bootstrap_CI95"] = {c: [round(BT[c].quantile(0.025), 4), round(BT[c].quantile(0.975), 4)] for c in BT}
R["point"] = dict(b_aod=B_AOD, b_soil=B_SOIL, pct_aod=100 * B_AOD / PV.ratio.mean(), pct_soil=100 * B_SOIL / PV.ratio.mean(), n=int(m0.nobs))
# attribution-share uncertainty: refit haze models on block-resampled haze days, PV coefficients drawn from bootstrap
sh = []
uH = np.unique(hblocks)
for b in range(200):
    hp = rng.choice(uH, len(uH), replace=True)
    Hbb = pd.concat([Hb[hblocks == k] for k in hp])
    Hm = Hbb.dropna(subset=FIRE + MET + ["pm25", "aod_ceres"])
    X = sm.add_constant(Hm[FIRE + MET]); mA = sm.OLS(Hm.aod_ceres, X).fit()
    r = BT.iloc[b % len(BT)]; tot = {}
    for c in C:
        cols = [x for x in FIRE if f"_{c}_" in x]; Z = Hm[FIRE + MET].copy()
        for cc in cols: Z[cc] = np.minimum(Z[cc], Hm[cc].quantile(0.10))
        dA = np.minimum((mA.predict(X) - mA.predict(sm.add_constant(Z, has_constant="add"))).clip(lower=0), (Hm.aod_ceres - AOD_BG).clip(lower=0))
        e = d.reindex(Hm.index)
        tot[c] = float((-r.b_aod * dA * e.e5_ssrd * (e.ceres_all / e.ceres_clr).clip(0, 1)).sum())
    s = sum(tot.values()); sh.append({c: 100 * tot[c] / s for c in C})
SH = pd.DataFrame(sh)
R["atm_share_CI95"] = {c: [round(SH[c].quantile(0.025), 1), round(SH[c].quantile(0.975), 1)] for c in C}
R["atm_share_point"] = {c: round(100 * base_att[c][0] / sum(base_att[k][0] for k in C), 1) for c in C}

# ---------------- PM2.5-dose-triggered cleaning optimisation ----------------
def simulate(policy, param=None, cost=None, rain_thr=2.0):
    """Replay 3 Mar 2023 - 7 Sep 2026 with observed PM2.5, rain and irradiance; soiling loss = -B_SOIL*dose*H*kWp."""
    idx = d.index[(d.index >= COMMISSION) & (d.index <= E.index.max())]; has = idx.isin(E.index) & d.pm25.reindex(idx).notna().values   # rev25: dose on every day, loss on fleet days
    pm = d.pm25.reindex(idx).fillna(0).values / 1000; tp = d.e5_tp.reindex(idx).fillna(0).values
    H = d.e5_ssrd.reindex(idx).fillna(0).values
    dose = 0.0; loss = 0.0; n = 0; last = 0
    for i, day in enumerate(idx):
        dose *= 1 - min(1.0, tp[i] / rain_thr)                       # rev25: partial rain cleaning
        clean = False
        if policy == "actual": clean = day in set(CLEAN + pd.Timedelta(days=1))   # rev25: as soil_dose
        elif policy == "calendar": clean = (i - last) >= param
        elif policy == "dose": clean = dose >= param
        if clean: dose = 0.0; n += 1; last = i
        dose += pm[i]
        loss += -B_SOIL * dose * H[i] * KWP_TOT * has[i]
    cc = CLEAN_COST if cost is None else cost
    return dict(cleanings=n, soiling_MWh=loss / 1000, cost_THB=n * cc + loss * TARIFF)
pol = {"actual (O&M log)": simulate("actual"), "none": simulate("none")}
for p in (60, 90, 120, 180): pol[f"calendar {p} d"] = simulate("calendar", p)
grid = np.round(np.arange(0.5, 6.01, 0.25), 2)
dose_res = {g: simulate("dose", g) for g in grid}
gbest = min(dose_res, key=lambda g: dose_res[g]["cost_THB"])
pol[f"dose-triggered D*={gbest} mg m-3 d"] = dose_res[gbest]
POL = pd.DataFrame(pol).T; POL["saving_vs_actual_THB"] = POL.loc["actual (O&M log)", "cost_THB"] - POL.cost_THB
POL.round(1).to_csv(f"{OUT}/T14_cleaning_policies.csv")
pd.DataFrame(dose_res).T.round(1).to_csv(f"{OUT}/T14b_dose_threshold_scan.csv")
R["cleaning"] = dict(D_star=float(gbest), **{k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in pol.items()})
# analytic check: renewal-reward optimum for constant dose rate r and irradiance H: D* = sqrt(2*Cc*r/(k*H*kWp*tariff))
r_dry = d.loc[d.index.month.isin([2, 3, 4]), "pm25"].mean() / 1000
H_dry = d.loc[d.index.month.isin([2, 3, 4]), "e5_ssrd"].mean()
R["cleaning"]["analytic_Dstar_dry"] = float(np.sqrt(2 * CLEAN_COST * r_dry / (-B_SOIL * H_dry * KWP_TOT * TARIFF)))
R["cleaning"]["analytic_interval_days_dry"] = float(R["cleaning"]["analytic_Dstar_dry"] / r_dry)

# ---------------- scenarios ----------------
Hs = haze_frame(TFE["p850"])
sc = {}
for name, cs, f in (("Thai fires -50%", ["THA"], 0.5), ("All fires -50%", C, 0.5), ("Myanmar fires -50%", ["MMR"], 0.5)):
    Hm = Hs.dropna(subset=FIRE + MET + ["pm25", "aod_ceres"]); X = sm.add_constant(Hm[FIRE + MET])
    Z = Hm[FIRE + MET].copy()
    for c in cs:
        for L in (0, 1): Z[f"lt_{c}_{L}"] = np.log1p(np.expm1(Z[f"lt_{c}_{L}"]) * f)
    dA = (mA0.predict(X) - mA0.predict(sm.add_constant(Z, has_constant="add"))).clip(lower=0)
    dP = Hm.pm25 * (1 - np.exp(-(mP0.predict(X) - mP0.predict(sm.add_constant(Z, has_constant="add"))).clip(lower=0)))
    e = d.reindex(Hm.index)
    atm = (-B_AOD * dA * e.e5_ssrd * (e.ceres_all / e.ceres_clr).clip(0, 1) * KWP_TOT * (Hm.index >= COMMISSION)).groupby(Hm.index.year).sum() / 1000
    sc[name] = dict(MWh_per_season=float(atm.mean()), THB=float(atm.mean() * 1000 * TARIFF), tCO2=float(atm.mean() * EF),
                    PM_drop_ugm3=float(dP.mean()), days_gt37_5_avoided=float(((Hm.pm25 > 37.5) & (Hm.pm25 - dP <= 37.5)).groupby(Hm.index.year).sum().mean()))
R["scenarios"] = sc
pd.DataFrame(sc).T.round(2).to_csv(f"{OUT}/T15_fire_reduction_scenarios.csv")

# season x ENSO
s = d[d.index.month.isin([3, 4])]
ens = s.groupby(s.index.year).agg(pm25=("pm25", "mean"), aod=("aod_ceres", "mean"), tfe_MMR=("tfe_MMR", "sum"), tfe_THA=("tfe_THA", "sum"),
                                 rain=("e5_tp", "sum"), oni=("oni", "mean"))
ens["atm_MWh"] = E.atm[E.index.month.isin([3, 4])].groupby(E.index[E.index.month.isin([3, 4])].year).sum() / 1000
ens["dep_MWh"] = E.dep[E.index.month.isin([3, 4])].groupby(E.index[E.index.month.isin([3, 4])].year).sum() / 1000
ens.round(2).to_csv(f"{OUT}/T16_MarApr_ENSO.csv")
yr.round(1).to_csv(f"{OUT}/T11b_loss_by_year.csv")

# cost sensitivity of the dose-triggered rule (break-even wash cost)
cs_rows = []
for rain_thr in (2.0, 5.0):
    for cost in (5_000, 10_000, 20_000, 30_000, 50_000, 100_000):
        res = {g: simulate("dose", g, cost, rain_thr) for g in np.round(np.arange(0.25, 4.01, 0.25), 2)}
        g = min(res, key=lambda k: res[k]["cost_THB"]); nn = simulate("none", None, cost, rain_thr); ac = simulate("actual", None, cost, rain_thr)
        cs_rows.append(dict(rain_reset_mm=rain_thr, wash_cost_THB=cost, D_star=g, washes=res[g]["cleanings"], soiling_MWh=round(res[g]["soiling_MWh"], 1),
                            total_cost_THB=round(res[g]["cost_THB"]), saving_vs_actual_THB=round(ac["cost_THB"] - res[g]["cost_THB"]),
                            saving_vs_none_THB=round(nn["cost_THB"] - res[g]["cost_THB"])))
CS = pd.DataFrame(cs_rows); CS.to_csv(f"{OUT}/T14c_cleaning_cost_sensitivity.csv", index=False); print(CS.to_string())
print(json.dumps(R, indent=1, default=float))
print(POL.round(1).to_string()); print(ens.round(2).to_string()); print(yr.round(1).to_string())
