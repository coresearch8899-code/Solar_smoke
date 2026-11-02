"""Block-length sensitivity (7, 14, 28 days) of the joint bootstrap, run with the procedure of joint_bootstrap_total_loss_rev25.py (same
clear-day sample, annual exposure sums and seed for every block length), so that the 14-day row reproduces Supplementary Table S7b.
Also: the atmospheric fraction of the total loss per replicate, and the wash-gain intervals of Table S2 (two definitions).
Usage: block_length_joint_rev25.py <daily_csv> <out_dir> <joint_script> <wash_csv>"""
import sys, json, numpy as np, pandas as pd
D3, OUT, JS, WCSV = sys.argv[1:5]
src = open(JS, encoding="utf-8").read(); src = src[:src.index("point_lin, point_hin = losses(PV)")]
sys.argv = [JS, D3, OUT]; G = {}; exec(compile(src, JS, "exec"), G)
PV, X, S_atm_lin, S_dep, B = G["PV"], G["X"], G["S_atm_lin"], G["S_dep"], 1000
sm = G["sm"]
def lin(df):
    m = sm.OLS(df.ratio, X(df, df.aod)).fit(); a, s = -m.params.aod * S_atm_lin, -m.params.soil * S_dep
    return dict(bA=m.params.aod, bS=m.params.soil, atm=a, dep=s, tot=a + s)
q = lambda s: [round(float(s.quantile(0.025)), 4), round(float(s.quantile(0.975)), 4)]
R = dict(point=lin(PV), blocks={})
for L in (7, 14, 28):
    blk = (PV.index - PV.index.min()).days // L; ub = np.unique(blk); rng = np.random.default_rng(42); groups = {k: PV[blk == k] for k in ub}
    O = pd.DataFrame([lin(pd.concat([groups[k] for k in rng.choice(ub, len(ub), replace=True)])) for _ in range(B)])
    R["blocks"][L] = dict(bA=q(O.bA), bS=q(O.bS), atm=q(O.atm), dep=q(O.dep), tot=q(O.tot), atm_frac_pct=q(100 * O.atm / O.tot))
# wash events: (i) measurement-only interval from the event SEs; (ii) between-event interval (t, n - 1 df)
from scipy import stats
W = pd.read_csv(WCSV).dropna(subset=["observed_gain_pct"]); n = len(W); g = W.observed_gain_pct
se_meas = float(np.sqrt((W.se_pct ** 2).sum()) / n); se_betw = float(g.std(ddof=1) / np.sqrt(n)); t = stats.t.ppf(0.975, n - 1)
R["wash"] = dict(n=n, mean=float(g.mean()), se_measurement=se_meas, ci_measurement=[float(g.mean() - 1.96 * se_meas), float(g.mean() + 1.96 * se_meas)],
                 se_between=se_betw, ci_between=[float(g.mean() - t * se_betw), float(g.mean() + t * se_betw)])
json.dump(R, open(f"{OUT}/block_length_joint.json", "w"), indent=1, default=float); print(json.dumps(R, indent=1, default=float))
