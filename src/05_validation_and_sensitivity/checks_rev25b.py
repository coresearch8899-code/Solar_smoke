"""rev25 second-round checks (pre-submission review 2):
A  bootstrap CI of the TOTAL (atmospheric + deposition) Shapley share by country, the quantity reported as the point estimate in Table 4.
   Each replicate refits the two haze models on 14-day blocks of haze days and draws the PV coefficients from a 14-day block bootstrap of the
   clear-day model; coalition values are then evaluated on the observed haze days. 1,000 replicates, with the CI at 200/500/1,000 for stability.
   The atmospheric-only share is reported with the same replicates.
B  nested leave-one-year-out validation: the PM2.5 rain rule is selected by AIC on the training years only, then the held-out year is predicted;
   absolute RMSE and out-of-sample R2 for the no-soiling, fixed partial-cleaning and nested-selected models.
C  Table 6: all washing policies at the same wash costs (10, 30, 100 kTHB), D* selected on the full record (in-sample).
Usage: checks_rev25b.py <step3_csv> <step3_dir_with_tfe> <out_dir>"""
import sys, json, math, itertools, warnings, numpy as np, pandas as pd, statsmodels.api as sm
warnings.filterwarnings("ignore")
D3, TFE_DIR, OUT = sys.argv[1:4]
KWP = {"Z1": 2256.24, "Z2": 207.06, "Z3": 49.90}; KWP_TOT = sum(KWP.values()); TARIFF = 4.2
CLEAN = pd.to_datetime(["2023-05-06", "2024-03-09", "2024-06-23", "2025-04-25", "2025-12-26", "2026-03-23", "2026-09-05"])
COMMISSION = pd.Timestamp("2023-03-03"); C = ["THA", "MMR", "LAO", "OTH"]
d = pd.read_csv(D3, index_col=0, parse_dates=True)
AOD_BG = d.loc[d.index.month.isin([11, 12, 1]), "aod_ceres"].median(); CLRW = (d.ceres_all / d.ceres_clr).clip(0, 1)
WASH = d.index.isin(CLEAN + pd.Timedelta(days=1)) | (d.index == COMMISSION)
def dose_partial(pm, pf=2.0):
    keep = 1 - np.minimum(1.0, d.e5_tp.reindex(pm.index).fillna(0).values / pf); x = pm.values / 1000.0
    out = np.full(len(pm), np.nan); D = 0.0
    for i in range(len(pm)):
        D = 0.0 if WASH[i] else D * keep[i]
        if not np.isnan(x[i]): D += x[i]; out[i] = D
    return pd.Series(out, index=pm.index)
def dose_reset(pm, thr):
    reset = (d.e5_tp >= thr) | WASH
    return pm.groupby(reset.cumsum()).cumsum() / 1000.0
d["dose"] = dose_partial(d.pm25)
doy = d.index.dayofyear
def pv_frame(soil):
    rows = []
    for z in KWP:
        t = pd.DataFrame({"ratio": d[f"Y_{z}"] / d.e5_ssrd, "aod": d.aod_ceres, "t2m": d.e5_t2m, "soil": soil,
                          "clear": (d.e5_tcc_day < 30) & (d.ceres_all / d.ceres_clr > 0.85) & (d.e5_tp < 1),
                          "yr": (d.index - COMMISSION).days / 365.25, "zone": z}, index=d.index)
        med = t.ratio.median(); rows.append(t[t.clear & t.ratio.between(0.4 * med, 1.3 * med)])
    return pd.concat(rows).drop(columns="clear").dropna()
def pv_X(P, soil=True):
    X = pd.get_dummies(P.zone).astype(float)
    for z in KWP: X[f"tr_{z}"] = (P.zone == z) * P.yr
    X[["aod", "t2m"] + (["soil"] if soil else [])] = P[["aod", "t2m"] + (["soil"] if soil else [])]; return X
PV = pv_frame(d.dose); m0 = sm.OLS(PV.ratio, pv_X(PV)).fit()
rng = np.random.default_rng(2025); R = {}

# ---------------- A: total Shapley share CI ----------------
MET = ["lblh", "inv850_07", "e5_rh", "e5_ws", "e5_tp", "ws850", "e5_tcwv", "s1", "c1"]
tfe = pd.read_csv(f"{TFE_DIR}/tfe_p850.csv", index_col=0, parse_dates=True)
H = d.drop(columns=[f"tfe_{c}" for c in C], errors="ignore").join(tfe, how="inner"); H = H[H.index.month.isin([1, 2, 3, 4, 5])].copy()
H["lblh"] = np.log(H.e5_blh_day); hd = H.index.dayofyear; H["s1"], H["c1"] = np.sin(2 * np.pi * hd / 365.25), np.cos(2 * np.pi * hd / 365.25)
FIRE = []
for c in C:
    for L in (0, 1): H[f"lt_{c}_{L}"] = np.log1p(H[f"tfe_{c}"].shift(L)); FIRE.append(f"lt_{c}_{L}")
H = H.dropna(subset=FIRE + MET + ["pm25", "aod_ceres"])
X0 = sm.add_constant(H[FIRE + MET]); Q10 = {cc: H[cc].quantile(0.10) for cc in FIRE}
COAL = [tuple(S) for r in range(1, 5) for S in itertools.combinations(C, r)]
Zc = {}
for S in COAL:
    Z = H[FIRE + MET].copy()
    for cc in [x for x in FIRE if x.split("_")[1] in S]: Z[cc] = np.minimum(Z[cc], Q10[cc])
    Zc[S] = sm.add_constant(Z, has_constant="add")
e = d.reindex(H.index); EXC = (H.aod_ceres - AOD_BG).clip(lower=0); W_ATM = e.e5_ssrd * CLRW.reindex(H.index).values * KWP_TOT / 1000 * (H.index >= COMMISSION)   # rev25: from commissioning
DEP_MASK = d.index.month.isin([1, 2, 3, 4, 5, 6]) & (d.index >= COMMISSION)
def unit_values(bA_model, bP_model):
    """Per coalition: atmospheric and deposition loss reduction per unit (-beta_A) and (-beta_S), MWh."""
    pA, pP = bA_model.predict(X0), bP_model.predict(X0); out = {}
    for S in COAL:
        dA = np.minimum((pA - bA_model.predict(Zc[S])).clip(lower=0), EXC)          # Eq. (8) with coalition S, daily clipping
        dP = H.pm25 * (1 - np.exp(-(pP - bP_model.predict(Zc[S])).clip(lower=0)))
        pm_c = d.pm25.copy(); pm_c.loc[H.index] = pm_c.loc[H.index] - dP
        dep = ((d.dose - dose_partial(pm_c)) * d.e5_ssrd * KWP_TOT / 1000)[DEP_MASK].sum()
        out[S] = (float((dA * W_ATM).sum()), float(dep))
    return out
def shapley_from(U, bA, bS, part=None):
    def v(S):
        if not S: return 0.0
        a, dp = U[tuple(sorted(S, key=C.index))]
        return {None: -bA * a - bS * dp, "atm": -bA * a, "dep": -bS * dp}[part]
    n = len(C); phi = {}
    for c in C:
        oth = [x for x in C if x != c]; s = 0.0
        for r in range(n):
            for S in itertools.combinations(oth, r):
                s += math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n) * (v(S + (c,)) - v(S))
        phi[c] = s
    return phi, v(tuple(C))
mA0 = sm.OLS(H.aod_ceres, X0).fit(); mP0 = sm.OLS(np.log(H.pm25), X0).fit(); U0 = unit_values(mA0, mP0)
phi0, j0 = shapley_from(U0, m0.params.aod, m0.params.soil); pa0, ja0 = shapley_from(U0, m0.params.aod, m0.params.soil, "atm")
R["A_point_total_share"] = {c: 100 * phi0[c] / j0 for c in C}; R["A_point_atm_share"] = {c: 100 * pa0[c] / ja0 for c in C}
R["A_point_MWh"] = dict(total={c: phi0[c] for c in C}, joint=j0)
hbl = (H.index - H.index.min()).days // 14; uh = np.unique(hbl)
pbl = (PV.index - PV.index.min()).days // 14; up = np.unique(pbl); PVg = {k: PV[pbl == k] for k in up}; Hg = {k: H[hbl == k] for k in uh}
rows = []
for b in range(1000):
    Hr = pd.concat([Hg[k] for k in rng.choice(uh, len(uh), replace=True)])
    Xr = sm.add_constant(Hr[FIRE + MET]); mA = sm.OLS(Hr.aod_ceres, Xr).fit(); mP = sm.OLS(np.log(Hr.pm25), Xr).fit()
    Pb = pd.concat([PVg[k] for k in rng.choice(up, len(up), replace=True)]); mb = sm.OLS(Pb.ratio, pv_X(Pb)).fit()
    U = unit_values(mA, mP); ph, jj = shapley_from(U, mb.params.aod, mb.params.soil); pa, ja = shapley_from(U, mb.params.aod, mb.params.soil, "atm")
    rows.append({**{f"tot_{c}": 100 * ph[c] / jj for c in C}, **{f"atm_{c}": 100 * pa[c] / ja for c in C}, "joint": jj})
BT = pd.DataFrame(rows); BT.to_csv(f"{OUT}/shapley_total_share_bootstrap.csv", index=False)
q = lambda s: [float(s.quantile(0.025)), float(s.quantile(0.975))]
R["A_CI_total_share"] = {c: q(BT[f"tot_{c}"]) for c in C}; R["A_CI_atm_share"] = {c: q(BT[f"atm_{c}"]) for c in C}
R["A_CI_total_share_by_n"] = {n: {c: q(BT[f"tot_{c}"].iloc[:n]) for c in C} for n in (200, 500, 1000)}
R["A_CI_joint_MWh"] = q(BT.joint)

# ---------------- B: nested LOYO validation ----------------
CAND = {"partial 2 mm": dose_partial(d.pm25, 2.0), "partial 5 mm": dose_partial(d.pm25, 5.0), "reset 0.5 mm": dose_reset(d.pm25, 0.5),
        "reset 2 mm": dose_reset(d.pm25, 2.0), "reset 5 mm": dose_reset(d.pm25, 5.0)}
FR = {k: pv_frame(s) for k, s in CAND.items()}
yrs = sorted(set(PV.index.year)); recs = []; err = {"none": [], "fixed": [], "nested": []}; obs = []
for y in yrs:
    best = None
    for k, P in FR.items():
        tr = P[P.index.year != y]; m = sm.OLS(tr.ratio, pv_X(tr)).fit()
        if best is None or m.aic < best[0]: best = (m.aic, k, m)
    _, kb, mb = best; te = FR[kb][FR[kb].index.year == y]; err["nested"].append(te.ratio - mb.predict(pv_X(te).reindex(columns=mb.params.index, fill_value=0)))
    P = FR["partial 2 mm"]; tr, te = P[P.index.year != y], P[P.index.year == y]
    mf = sm.OLS(tr.ratio, pv_X(tr)).fit(); err["fixed"].append(te.ratio - mf.predict(pv_X(te).reindex(columns=mf.params.index, fill_value=0)))
    mn = sm.OLS(tr.ratio, pv_X(tr, soil=False)).fit(); err["none"].append(te.ratio - mn.predict(pv_X(te, soil=False).reindex(columns=mn.params.index, fill_value=0)))
    obs.append(te.ratio); recs.append(dict(year=y, selected=kb, n=len(te)))
o = pd.concat(obs); sst = ((o - o.mean()) ** 2).sum()
def r2(e_): return float(1 - (e_ ** 2).sum() / sst)
B = {k: dict(rmse=float(np.sqrt((pd.concat(v) ** 2).mean())), r2_oos=r2(pd.concat(v))) for k, v in err.items()}
for k in B: B[k]["by_year_rmse"] = {int(y): float(np.sqrt((v_ ** 2).mean())) for y, v_ in zip(yrs, err[k])}
R["B_nested_loyo"] = dict(models=B, selected_by_year=recs, mean_ratio=float(o.mean()), sd_ratio=float(o.std()))

# ---------------- C: Table 6 at common wash costs ----------------
E_idx = d.index[(sum(d[f"Y_{z}"] * k for z, k in KWP.items())).notna()]
ALL = d.index[(d.index >= COMMISSION) & (d.index <= E_idx.max())]; HAS = ALL.isin(E_idx) & d.pm25.reindex(ALL).notna().values   # rev25: dose on every day, loss on fleet days
pm = d.pm25.reindex(ALL).fillna(0).values / 1000; tp = d.e5_tp.reindex(ALL).fillna(0).values; Hh = d.e5_ssrd.reindex(ALL).fillna(0).values
BS = m0.params.soil
def replay(policy, param=None, pf=2.0):
    D = 0.0; loss = 0.0; n = 0; last = 0; cl = set(CLEAN + pd.Timedelta(days=1))
    for i, day in enumerate(ALL):
        D *= 1 - min(1.0, tp[i] / pf)
        if (policy == "actual" and day in cl) or (policy == "calendar" and i - last >= param) or (policy == "dose" and D >= param): D = 0.0; n += 1; last = i
        D += pm[i]; loss += -BS * D * Hh[i] * KWP_TOT * HAS[i]
    return n, loss / 1000
rowsC = {"Actual O&M log": replay("actual"), "No washing": replay("none")}
for k in (60, 120, 180): rowsC[f"Calendar, every {k} d"] = replay("calendar", k)
GRID = np.round(np.arange(0.25, 4.01, 0.25), 2); scan = {g: replay("dose", g) for g in GRID}
T6 = []
for name, (n, l) in rowsC.items():
    T6.append(dict(policy=name, washes=n, soiling_MWh=l, **{f"cost_{c}": n * c + l * TARIFF for c in (10, 30, 100)}))
dose = {}
for c in (10, 30, 100):
    g = min(GRID, key=lambda g: scan[g][0] * c + scan[g][1] * TARIFF); n, l = scan[g]
    dose[c] = dict(Dstar=float(g), washes=n, soiling_MWh=l, cost=n * c + l * TARIFF)
R["C_table6_fixed"] = T6; R["C_table6_dose"] = dose
json.dump(R, open(f"{OUT}/checks_rev25b.json", "w"), indent=1, default=float)
print(json.dumps({k: R[k] for k in R if not k.startswith("A_CI_total_share_by")}, indent=1, default=float)); print(json.dumps(R["A_CI_total_share_by_n"], indent=1))
