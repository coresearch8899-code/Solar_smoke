"""rev25 (partial-cleaning dose). Step 5 – revision analyses (reviewer-proofing). Python only; inputs are the step-3 daily table and TFE files.
R1 single-pipeline loss table (Table 5)        R2 exact break-even wash cost (Table 6, Fig 6b)
R3 Shapley (additive) country attribution        R4 placebo tests (lead exposure; year-swapped exposure)
R5 alternative exposure forms (linear, sqrt, near/far)
R6 hinge (threshold) AOD response and its effect on losses
R7 wash-event validation of the soiling model     R8 on-site pyranometer around washes
R9 'Other' exposure decline (trajectory origins)"""
import sys, json, itertools, math, warnings, numpy as np, pandas as pd, statsmodels.api as sm
warnings.filterwarnings("ignore")
D3, TFE_DIR, OUT = sys.argv[1:4]
KWP = {"Z1": 2256.24, "Z2": 207.06, "Z3": 49.90}; KWP_TOT = sum(KWP.values())
CLEAN = pd.to_datetime(["2023-05-06", "2024-03-09", "2024-06-23", "2025-04-25", "2025-12-26", "2026-03-23", "2026-09-05"])
TARIFF, EF = 4.2, 0.4857
C = ["THA", "MMR", "LAO", "OTH"]; R = {}
hac = dict(cov_type="HAC", cov_kwds={"maxlags": 7})
d = pd.read_csv(D3, index_col=0, parse_dates=True)
AOD_BG = d.loc[d.index.month.isin([11, 12, 1]), "aod_ceres"].median()
CLRW = (d.ceres_all / d.ceres_clr).clip(0, 1)

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

# ---------------- PV frame (identical to step 4) ----------------
rows = []
for z in KWP:
    t = pd.DataFrame({"ratio": d[f"Y_{z}"] / d.e5_ssrd, "aod": d.aod_ceres, "t2m": d.e5_t2m, "soil": d.dose,
                      "clear": (d.e5_tcc_day < 30) & (d.ceres_all / d.ceres_clr > 0.85) & (d.e5_tp < 1),
                      "yr": (d.index - pd.Timestamp("2023-03-03")).days / 365.25, "zone": z}, index=d.index)
    med = t.ratio.median(); t["ok"] = t.ratio.between(0.4 * med, 1.3 * med); rows.append(t)
ALL = pd.concat(rows); PV = ALL[ALL.clear & ALL.ok].drop(columns=["clear", "ok"]).dropna()

def pv_X(df, aod_cols=("aod",)):
    X = pd.get_dummies(df.zone).astype(float)
    for z in KWP: X[f"tr_{z}"] = (df.zone == z) * df.yr
    for a in aod_cols: X[a] = df[a]
    X[["t2m", "soil"]] = df[["t2m", "soil"]]
    return X
m0 = sm.OLS(PV.ratio, pv_X(PV)).fit(); B_AOD, B_SOIL = m0.params.aod, m0.params.soil
MEAN_RATIO = PV.ratio.mean()

# ================= R1 single-pipeline annual table =================
E = pd.DataFrame(index=d.index)
E["atm"] = -B_AOD * (d.aod_ceres - AOD_BG).clip(lower=0) * d.e5_ssrd * CLRW * KWP_TOT
E["dep"] = -B_SOIL * d.dose * d.e5_ssrd * KWP_TOT
E["act"] = sum(d[f"Y_{z}"] * k for z, k in KWP.items())
E = E[E.act.notna()]
jm = E.index.month.isin([1, 2, 3, 4, 5])
T5 = (E.groupby(E.index.year)[["act", "atm", "dep"]].sum() / 1000).rename(columns={"act": "E_actual_MWh", "atm": "atm_MWh", "dep": "dep_MWh"})
T5["days"] = E.groupby(E.index.year).size()
T5["loss_pct"] = 100 * (T5.atm_MWh + T5.dep_MWh) / (T5.E_actual_MWh + T5.atm_MWh + T5.dep_MWh)
T5["atm_JanMay"] = E[jm].groupby(E.index[jm].year).atm.sum() / 1000
T5["dep_JanMay"] = E[jm].groupby(E.index[jm].year).dep.sum() / 1000
T5.round(1).to_csv(f"{OUT}/R1_table5_single_pipeline.csv")
full = T5.loc[[2024, 2025]]
R["R1_annual_2024_2025"] = dict(atm=full.atm_MWh.mean(), dep=full.dep_MWh.mean(), tot=(full.atm_MWh + full.dep_MWh).mean(),
                               loss_pct=full.loss_pct.mean(), THB=(full.atm_MWh + full.dep_MWh).mean() * 1000 * TARIFF,
                               tCO2=(full.atm_MWh + full.dep_MWh).mean() * EF)
R["R1_days"] = dict(calendar_days=int((pd.Timestamp("2026-09-07") - pd.Timestamp("2023-03-03")).days + 1), days_with_fleet_data=int(len(E)))

# ================= R2 washing replay: exact break-even =================
def replay(policy, param=None, rain_thr=2.0):
    idx = d.index[(d.index >= COMMISSION) & (d.index <= E.index.max())]; has = idx.isin(E.index) & d.pm25.reindex(idx).notna().values   # rev25: dose on every day, loss on fleet days
    pm = d.pm25.reindex(idx).fillna(0).values / 1000; tp = d.e5_tp.reindex(idx).fillna(0).values
    H = d.e5_ssrd.reindex(idx).fillna(0).values; dose = 0.0; loss = 0.0; n = 0
    cl = set(CLEAN + pd.Timedelta(days=1))   # rev25: as soil_dose
    for i, day in enumerate(idx):
        dose *= 1 - min(1.0, tp[i] / rain_thr)                       # rev25: partial rain cleaning
        if (policy == "actual" and day in cl) or (policy == "dose" and dose >= param): dose = 0.0; n += 1
        dose += pm[i]; loss += -B_SOIL * dose * H[i] * KWP_TOT * has[i]
    return n, loss / 1000
be = {}; scan_rows = []
for thr in (2.0, 5.0):
    n0, l0 = replay("none", rain_thr=thr); na, la = replay("actual", rain_thr=thr)
    best_c = 0.0
    for D in np.round(np.arange(0.25, 4.01, 0.25), 2):
        n, l = replay("dose", D, thr)
        c_be = (l0 - l) * 1000 * TARIFF / n if n else np.nan                # wash cost at which this rule just breaks even
        scan_rows.append(dict(rain_reset_mm=thr, D_star=D, washes=n, soiling_MWh=round(l, 1), breakeven_THB=round(c_be) if n else None))
        if n and c_be > best_c: best_c = c_be
    be[f"{thr:g}mm"] = dict(no_wash_MWh=round(l0, 1), no_wash_cost_THB=round(l0 * 1000 * TARIFF), actual_MWh=round(la, 1),
                           breakeven_wash_cost_THB=round(best_c))
pd.DataFrame(scan_rows).to_csv(f"{OUT}/R2_dose_rule_breakeven_scan.csv", index=False)
R["R2_breakeven"] = be

# ================= haze-model machinery (as in step 4, generalised) =================
TFE = {k: pd.read_csv(f"{TFE_DIR}/tfe_{k}.csv", index_col=0, parse_dates=True) for k in ("p850", "p700", "3d850")}
MET = ["lblh", "inv850_07", "e5_rh", "e5_ws", "e5_tp", "ws850", "e5_tcwv", "s1", "c1"]
TRANS = {"log": np.log1p, "sqrt": np.sqrt, "linear": lambda x: x / 1000.0}
def haze_frame(tfe, form="log", nearfar=False):
    H = d.drop(columns=[f"tfe_{c}" for c in C]).join(tfe, how="inner")
    H = H[H.index.month.isin([1, 2, 3, 4, 5])].copy()
    H["lblh"] = np.log(H.e5_blh_day); doy = H.index.dayofyear
    H["s1"], H["c1"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    fire = []
    for c in C:
        parts = {"": H[f"tfe_{c}"]} if not nearfar else {"near": H[f"tfe24_{c}"], "far": H[f"tfe_{c}"] - H[f"tfe24_{c}"]}
        for tag, s in parts.items():
            for L in (0, 1):
                col = f"lt_{c}_{L}{tag}"; H[col] = TRANS[form](s.shift(L)); fire.append(col)
    return H.dropna(subset=fire + MET + ["pm25", "aod_ceres"]), fire

def att_engine(H, fire, hinge=None, atm_only=False):
    X = sm.add_constant(H[fire + MET]); mA = sm.OLS(H.aod_ceres, X).fit(); mP = sm.OLS(np.log(H.pm25), X).fit()
    bA, bP = mA.predict(X), mP.predict(X); e = d.reindex(H.index)
    b_soil = B_SOIL if hinge is None else hinge[2]
    def v(coal, by_year=False):
        if not coal: return (0.0, 0.0) if not by_year else None
        Z = H[fire + MET].copy()
        for cc in [x for x in fire if x.split("_")[1] in coal]: Z[cc] = np.minimum(Z[cc], H[cc].quantile(0.10))
        Zc = sm.add_constant(Z, has_constant="add")
        dA = np.minimum((bA - mA.predict(Zc)).clip(lower=0), (H.aod_ceres - AOD_BG).clip(lower=0))
        dP = H.pm25 * (1 - np.exp(-(bP - mP.predict(Zc)).clip(lower=0)))
        if hinge is None: dlin = dA
        else: dlin = (H.aod_ceres - hinge[0]).clip(lower=0) - (H.aod_ceres - dA - hinge[0]).clip(lower=0)
        b_aod = B_AOD if hinge is None else hinge[1]
        atm = -b_aod * dlin * e.e5_ssrd * CLRW.reindex(H.index).values * KWP_TOT / 1000 * (H.index >= COMMISSION)   # rev25: from commissioning
        if atm_only: return float(atm.sum()), 0.0
        pm_c = d.pm25.copy(); pm_c.loc[H.index] = pm_c.loc[H.index] - dP
        dep = -b_soil * (d.dose - soil_dose(pm_c, d.e5_tp)) * d.e5_ssrd * KWP_TOT / 1000
        dep = dep[dep.index.month.isin([1, 2, 3, 4, 5, 6]) & (dep.index >= COMMISSION)]
        if by_year: return atm.groupby(atm.index.year).sum(), dep.groupby(dep.index.year).sum()
        return float(atm.sum()), float(dep.sum())
    return v, mA, mP

def shapley(v, part=None):
    """Shapley value of each country; part=0 atmospheric, 1 deposition, None both."""
    n = len(C); cache = {}
    def val(S):
        k = frozenset(S)
        if k not in cache:
            r = v(tuple(sorted(k))); cache[k] = sum(r) if part is None else r[part]
        return cache[k]
    phi = {c: 0.0 for c in C}
    for c in C:
        others = [x for x in C if x != c]
        for r in range(n):
            for S in itertools.combinations(others, r):
                w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                phi[c] += w * (val(S + (c,)) - val(S))
    return phi, val(tuple(C))

def single_shares(v):
    tot = {c: sum(v((c,))) for c in C}; s = sum(tot.values()); return {c: 100 * tot[c] / s for c in C}

# ================= R3 Shapley attribution by set-up =================
R3 = {}
for k in ("p850", "p700", "3d850"):
    H, fire = haze_frame(TFE[k]); v, _, _ = att_engine(H, fire)
    phi, joint = shapley(v); pa, ja = shapley(v, 0); pdp, jd = shapley(v, 1)
    R3[k] = dict(shapley_MWh={c: round(phi[c], 1) for c in C}, joint_all_MWh=round(joint, 1),
                 shapley_atm_MWh={c: round(pa[c], 1) for c in C}, shapley_dep_MWh={c: round(pdp[c], 1) for c in C},
                 joint_atm_MWh=round(ja, 1), joint_dep_MWh=round(jd, 1),
                 shapley_share={c: round(100 * phi[c] / joint, 1) for c in C},
                 single_share={c: round(x, 1) for c, x in single_shares(v).items()},
                 exposure_share={c: round(100 * TFE[k][f"tfe_{c}"].sum() / TFE[k][[f"tfe_{x}" for x in C]].sum().sum(), 1) for c in C})
R["R3_shapley"] = R3
# block-bootstrap CI (14-day blocks, 200 replicates) of the Shapley atmospheric share, 850 hPa
# (the PV coefficient scales all countries equally, so it cancels in the shares)
rng = np.random.default_rng(42)
Hb, Fb = haze_frame(TFE["p850"]); hbl = (Hb.index - Hb.index.min()).days // 14; ub = np.unique(hbl); bs = []
for b in range(200):
    Hr = pd.concat([Hb[hbl == k] for k in rng.choice(ub, len(ub), replace=True)])
    vb, _, _ = att_engine(Hr, Fb, atm_only=True); pb, jb = shapley(vb, 0); bs.append({c: 100 * pb[c] / jb for c in C})
BS = pd.DataFrame(bs); R["R3_shapley_atm_share_CI95_p850"] = {c: [round(BS[c].quantile(0.025), 1), round(BS[c].quantile(0.975), 1)] for c in C}
# per-year Shapley-consistent split for Fig 6a (850 hPa): Thai vs abroad, scaled to the joint per-year total
H850, F850 = haze_frame(TFE["p850"]); v850, mA850, mP850 = att_engine(H850, F850)
ya, yd = v850(tuple(C), by_year=True)
sh = R3["p850"]["shapley_share"]
fig6 = pd.DataFrame({"fire_atm_MWh": ya, "fire_dep_MWh": yd}).fillna(0)
fig6["thai_atm_MWh"] = fig6.fire_atm_MWh * sh["THA"] / 100; fig6["abroad_atm_MWh"] = fig6.fire_atm_MWh * (100 - sh["THA"]) / 100
fig6 = fig6.join(T5[["atm_JanMay", "dep_JanMay"]]); fig6["nonfire_atm_MWh"] = fig6.atm_JanMay - fig6.fire_atm_MWh
fig6.round(2).to_csv(f"{OUT}/R3_fig6a_components.csv")
R["R3_fire_share_of_JanMay_atm_pct"] = round(100 * fig6.fire_atm_MWh.sum() / fig6.atm_JanMay.sum(), 1)

# ================= R4 placebo tests =================
X_met = sm.add_constant(H850[MET])
pl = {}
for yname, y in (("logPM25", np.log(H850.pm25)), ("AOD", H850.aod_ceres)):
    met_aic = sm.OLS(y, X_met).fit().aic
    real = sm.OLS(y, sm.add_constant(H850[F850 + MET])).fit(**hac)
    # (a) lead exposure (2 days ahead) added to the full model: should carry no signal
    Hl = H850.copy(); lead = []
    for c in C:
        Hl[f"lead2_{c}"] = np.log1p(TFE["p850"][f"tfe_{c}"].reindex(Hl.index + pd.Timedelta(days=2)).values); lead.append(f"lead2_{c}")
    Hl = Hl.dropna(subset=lead)
    ml = sm.OLS(y.reindex(Hl.index), sm.add_constant(Hl[F850 + lead + MET])).fit(**hac)
    # (b) year-swapped exposure: same calendar day, other year (keeps seasonality, breaks day-to-day transport link)
    years = sorted(set(H850.index.year)); swaps = []
    for perm in itertools.permutations(years):
        if any(a == b for a, b in zip(years, perm)): continue                # derangements only
        Hs = H850.copy(); mp = dict(zip(years, perm))
        src = [pd.Timestamp(year=mp[t.year], month=t.month, day=min(t.day, 28 if t.month == 2 else t.day)) for t in Hs.index]
        for col in F850:
            c, L = col.split("_")[1], int(col.split("_")[2])
            Hs[col] = np.log1p(TFE["p850"][f"tfe_{c}"].reindex(pd.DatetimeIndex(src) - pd.Timedelta(days=L)).values)
        Hs = Hs.dropna(subset=F850)
        ms = sm.OLS(y.reindex(Hs.index), sm.add_constant(Hs[F850 + MET])).fit()
        swaps.append(ms.aic - sm.OLS(y.reindex(Hs.index), sm.add_constant(Hs[MET])).fit().aic)
    pl[yname] = dict(dAIC_real=round(real.aic - met_aic, 1),
                     lead2_coef={c: round(ml.params[f"lead2_{c}"], 4) for c in C}, lead2_p={c: round(ml.pvalues[f"lead2_{c}"], 3) for c in C},
                     lag0_THA_with_leads=round(ml.params["lt_THA_0"], 4), lag0_THA_with_leads_p=round(ml.pvalues["lt_THA_0"], 4),
                     dAIC_yearswap_mean=round(float(np.mean(swaps)), 1), dAIC_yearswap_range=[round(min(swaps), 1), round(max(swaps), 1)], n_swaps=len(swaps))
R["R4_placebo"] = pl

# ================= R5 alternative exposure forms =================
R5 = {}
for form in ("log", "sqrt", "linear"):
    H, fire = haze_frame(TFE["p850"], form); v, mA, _ = att_engine(H, fire); phi, joint = shapley(v)
    R5[form] = dict(aic_AOD=round(mA.aic, 1), shapley_share={c: round(100 * phi[c] / joint, 1) for c in C})
H, fire = haze_frame(TFE["p850"], "log", nearfar=True); v, mA, mP = att_engine(H, fire); phi, joint = shapley(v)
mPh = sm.OLS(np.log(H.pm25), sm.add_constant(H[fire + MET])).fit(**hac); mAh = sm.OLS(H.aod_ceres, sm.add_constant(H[fire + MET])).fit(**hac)
R5["near24h_far"] = dict(aic_AOD=round(mA.aic, 1), shapley_share={c: round(100 * phi[c] / joint, 1) for c in C},
                         coef_AOD={k: [round(mAh.params[k], 4), round(mAh.pvalues[k], 3)] for k in fire},
                         coef_logPM={k: [round(mPh.params[k], 4), round(mPh.pvalues[k], 3)] for k in fire},
                         near_share_of_TFE={c: round(100 * TFE["p850"][f"tfe24_{c}"].sum() / TFE["p850"][f"tfe_{c}"].sum(), 1) for c in C})
R["R5_exposure_forms"] = R5

# ================= R6 hinge AOD response =================
hs = []
for k in np.round(np.arange(0.0, 0.81, 0.05), 2):
    P_ = PV.assign(aod_h=(PV.aod - k).clip(lower=0)); mh = sm.OLS(P_.ratio, pv_X(P_, ("aod_h",))).fit(**hac)
    hs.append(dict(knot=k, aic=round(mh.aic, 1), b=mh.params.aod_h, se=mh.bse.aod_h, b_soil=mh.params.soil))
HS = pd.DataFrame(hs); HS.round(4).to_csv(f"{OUT}/R6_hinge_scan.csv", index=False)
kb = HS.loc[HS.aic.idxmin()]
atm_h = -kb.b * ((d.aod_ceres - kb.knot).clip(lower=0) - max(AOD_BG - kb.knot, 0)).clip(lower=0) * d.e5_ssrd * CLRW * KWP_TOT
atm_h = atm_h.reindex(E.index); dep_h = -kb.b_soil * d.dose.reindex(E.index) * d.e5_ssrd.reindex(E.index) * KWP_TOT
yh = pd.DataFrame({"atm": atm_h, "dep": dep_h}).groupby(E.index.year).sum() / 1000
fh = yh.loc[[2024, 2025]].mean()
yh["loss_pct"] = 100 * (yh.atm + yh.dep) / (T5.E_actual_MWh + yh.atm + yh.dep); yh.round(1).to_csv(f"{OUT}/R6_loss_by_year_hinge.csv")
R["R6_hinge"] = dict(knot=float(kb.knot), aic=float(kb.aic), aic_linear=float(HS.loc[HS.knot == 0, "aic"].iloc[0]), b=float(kb.b), se=float(kb.se),
                     pct_per_unit_aod_above_knot=float(100 * kb.b / MEAN_RATIO),
                     annual_2024_2025=dict(atm=float(fh.atm), dep=float(fh.dep), tot=float(fh.sum()),
                                           loss_pct=float(100 * fh.sum() / (T5.loc[[2024, 2025], "E_actual_MWh"].mean() + fh.sum()))))
R10 = {}
for k in ("p850", "p700", "3d850"):
    H, fire = haze_frame(TFE[k]); v, _, _ = att_engine(H, fire, hinge=(float(kb.knot), float(kb.b), float(kb.b_soil)))
    phi, joint = shapley(v); R10[k] = dict(joint_MWh=round(joint, 1), shapley_share={c: round(100 * phi[c] / joint, 1) for c in C})
R["R10_hinge_attribution"] = R10
# binned partial-residual data for Fig 4a (residual after zone, trend, T2m, soiling)
Xf = pv_X(PV); part = PV.ratio - (Xf.drop(columns=["aod"]) * m0.params.drop("aod")).sum(1)
pd.DataFrame({"aod": PV.aod, "partial": part / MEAN_RATIO}).to_csv(f"{OUT}/R6_fig4a_partial.csv")
R["R6_fig4a_line"] = dict(lin_a=float(((part - B_AOD * PV.aod)).mean() / MEAN_RATIO), lin_b=float(B_AOD / MEAN_RATIO))
mh_best = sm.OLS(PV.ratio, pv_X(PV.assign(aod_h=(PV.aod - kb.knot).clip(lower=0)), ("aod_h",))).fit()
Xh = pv_X(PV.assign(aod_h=(PV.aod - kb.knot).clip(lower=0)), ("aod_h",))
part_h = PV.ratio - (Xh.drop(columns=["aod_h"]) * mh_best.params.drop("aod_h")).sum(1)
# Fig 4a points: hinge-model partial residual, normalised so that days below the knot average 1
pd.DataFrame({"aod": PV.aod, "rel": 1 + (part_h - part_h[PV.aod <= kb.knot].mean()) / MEAN_RATIO}).to_csv(f"{OUT}/R6_fig4a_partial_hinge.csv")
R["R6_fig4a_hinge"] = dict(a=float(part_h[PV.aod <= kb.knot].mean() / MEAN_RATIO), b=float(kb.b / MEAN_RATIO), knot=float(kb.knot))

# ================= R7 wash-event validation =================
# residual ratio with the soiling term added back, on near-clear days (looser screen to keep days around washes)
A7 = ALL[ALL.ok].dropna(subset=["ratio", "aod", "t2m"]).copy()
A7["near_clear"] = (d.e5_tcc_day.reindex(A7.index) < 50) & ((d.ceres_all / d.ceres_clr).reindex(A7.index) > 0.75)
A7 = A7[A7.near_clear]
X7 = pv_X(A7); A7["res_nosoil"] = A7.ratio - (X7.drop(columns=["soil"]) * m0.params.drop("soil")).sum(1)
w7 = []
for w in CLEAN:
    b = A7[(A7.index >= w - pd.Timedelta(days=10)) & (A7.index < w)]; a = A7[(A7.index > w) & (A7.index <= w + pd.Timedelta(days=10))]
    if len(b) < 3 or len(a) < 3: w7.append(dict(wash=str(w.date()), n_before=len(b), n_after=len(a))); continue
    obs = 100 * (a.res_nosoil.mean() - b.res_nosoil.mean()) / MEAN_RATIO
    se = 100 * np.sqrt(a.res_nosoil.var() / len(a) + b.res_nosoil.var() / len(b)) / MEAN_RATIO
    pred = 100 * -B_SOIL * (b.soil.mean() - a.soil.mean()) / MEAN_RATIO
    rain = int((d.e5_tp.loc[w - pd.Timedelta(days=10): w + pd.Timedelta(days=10)] >= 2).sum())
    w7.append(dict(wash=str(w.date()), n_before=len(b), n_after=len(a), dose_before=round(b.soil.mean(), 2), observed_gain_pct=round(obs, 2),
                   se_pct=round(se, 2), predicted_gain_pct=round(pred, 2), rain_days_ge2mm_window=rain))
W7 = pd.DataFrame(w7); W7.to_csv(f"{OUT}/R7_wash_event_validation.csv", index=False)
v7 = W7.dropna(subset=["observed_gain_pct"])
R["R7_wash_validation"] = dict(n_events=int(len(v7)), mean_observed=round(v7.observed_gain_pct.mean(), 2), mean_predicted=round(v7.predicted_gain_pct.mean(), 2),
                              corr=round(float(np.corrcoef(v7.observed_gain_pct, v7.predicted_gain_pct)[0, 1]), 2) if len(v7) > 2 else None,
                              pooled_se=round(float(np.sqrt((v7.se_pct ** 2).sum()) / len(v7)), 2))

# ================= R8 pyranometer around washes / first valid day =================
s = d[(d.site_n_valid_ghi >= 80) & (d.e5_tcc_day < 30) & (d.aod_ceres < 0.4)]; rr = (s.site_ghi_int_kWh / s.e5_ssrd).dropna()
p8 = []
for w in CLEAN:
    b = rr[(rr.index >= w - pd.Timedelta(days=21)) & (rr.index < w)]; a = rr[(rr.index > w) & (rr.index <= w + pd.Timedelta(days=21))]
    p8.append(dict(wash=str(w.date()), n_before=len(b), n_after=len(a), jump_pct=round(100 * (a.mean() / b.mean() - 1), 1) if len(a) > 2 and len(b) > 2 else None))
pd.DataFrame(p8).to_csv(f"{OUT}/R8_pyranometer_wash_jumps.csv", index=False)
R["R8_pyranometer"] = dict(first_valid=str(rr.index.min().date()), last_valid=str(rr.index.max().date()), n=int(len(rr)), jumps=p8,
                           n_valid_days_all=int((d.site_n_valid_ghi >= 80).sum()), first_n_valid=str(d.index[d.site_n_valid_ghi >= 80].min().date()))

# ================= R9 'Other' exposure decline =================
t = TFE["p850"]; t = t[t.index.month.isin([1, 2, 3, 4, 5])]
R["R9_other"] = dict(frac_traj_from_China_north_of_23p5={int(y): round(100 * float(((g.end_lat > 23.5) & (g.end_lon > 98)).mean()), 1) for y, g in t.groupby(t.index.year)},
                     frac_traj_from_west_of_92E={int(y): round(100 * float((g.end_lon < 92).mean()), 1) for y, g in t.groupby(t.index.year)},
                     tfe_OTH_sum={int(y): round(float(g.tfe_OTH.sum())) for y, g in t.groupby(t.index.year)})

json.dump(R, open(f"{OUT}/R_revision_results.json", "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
print(T5.round(1).to_string()); print(W7.to_string()); print(HS.round(4).to_string())
