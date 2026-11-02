"""Rev25 figures: as figures_rev20.py (same data and layout, centred '(a)', '(b)', '(c)' labels, vector PDF + PNG + LZW TIFF), plus
- bitmaps written at >= 1000 dpi and >= 7480 px wide (Renewable Energy guide: full-page line art 7480 px at 1000 dpi),
- Figure 2a: north arrow and 200 km scale bar added to the map,
- Figure 6a: legend labels use a colon instead of an en dash between words.
Figure 2 cannot be redrawn here (the step-12 trajectory and fire files are not on this computer), so its rev19 vector PDF is edited:
the two panel titles are removed, centred '(a)'/'(b)' labels and the map furniture are inserted, then the page is rasterised."""
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager
import pymupdf
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
S3, S4, S5, S7, FIG2PDF, FD, ROB = sys.argv[1:8]          # ROB: robustness_rev25.py output (soiling-metric comparison)
DPI = 1000
B, Or, Aq, Y, V, INK, MUT, GR = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#0b0b0b", "#52514e", "#c3c2b7"
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans", "axes.edgecolor": GR, "xtick.color": MUT, "ytick.color": MUT, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9.5, "axes.titlelocation": "center"})
d = pd.read_csv(f"{S3}/CRH_haze_daily_step3.csv", index_col=0, parse_dates=True)
R = json.load(open(f"{S5}/R_revision_results.json"))
def grid(ax): ax.grid(axis="y", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
def label(ax, k): ax.set_title(f"({k})", loc="center", fontsize=9.5, pad=6)
MINPX = 7500                                   # > 7480 px, the full-page minimum for 1000-dpi line art
def dpi_for(width_in): return max(DPI, int(np.ceil(MINPX / width_in)))
def tif(png, name, dpi=DPI):
    im = Image.open(png).convert("RGB"); im.save(f"{FD}/{name}.tif", compression="tiff_lzw", dpi=(dpi, dpi))
def save(fig, name):
    dpi = dpi_for(fig.get_size_inches()[0])
    fig.savefig(f"{FD}/{name}.pdf"); fig.savefig(f"{FD}/{name}.png", dpi=dpi); plt.close(fig); tif(f"{FD}/{name}.png", name, dpi)
C = ["THA", "MMR", "LAO", "OTH"]; labs = ["Thailand", "Myanmar", "Laos", "Other"]

# ---------- Fig 1 framework (single panel, no panel label) ----------
fig, ax = plt.subplots(figsize=(7.4, 4.0)); ax.axis("off"); ax.set_xlim(0, 104); ax.set_ylim(0, 60)
def box(x, y, w, h, text, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=col, ec="none", alpha=0.18))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=5.6, color=INK)
cols = [("Open data", B, ["VIIRS active fires\n(S-NPP, NOAA-20/21)\n9.4 M detections", "ERA5 reanalysis\n(winds, irradiance, BLH)", "CAMS PM$_{2.5}$, AOD,\nCERES SYN1deg", "ESA WorldCover,\nGADM boundaries"]),
        ("Site data", Aq, ["2.51 MWp hospital\nrooftop PV, 3 zones\n1,283 days", "On-site meteo\nstation (15 min)", "O&M log:\n7 fleet washes"]),
        ("Methods", Y, ["72-h back-trajectories\n(850/700 hPa, 3-D)", "Trajectory fire\nexposure by country", "Haze models\n+ placebo tests", "Clear-day PV model:\nAOD + PM$_{2.5}$ dose"]),
        ("Outputs", Or, ["Atmospheric vs\ndeposition loss", "Shapley country\nattribution", "Dose-triggered\nwashing rule", "Fire-reduction\nscenarios"])]
for i, (title, col, items) in enumerate(cols):
    x = 2 + i * 26; ax.text(x + 10.5, 57, title, ha="center", fontsize=9, fontweight="bold", color=INK); hh = 52 / len(items)
    for j, it in enumerate(items): box(x, 53 - (j + 1) * hh + 1.2, 21, hh - 3.2, it, col)
    if i < 3: ax.annotate("", xy=(x + 25.3, 28), xytext=(x + 22.6, 28), arrowprops=dict(arrowstyle="-|>", color=MUT, lw=1.2))
save(fig, "Figure_1")

# ---------- Fig 2: edit the rev19 vector PDF ----------
doc = pymupdf.open(FIG2PDF); pg = doc[0]
font = font_manager.findfont("DejaVu Sans")
def centre_of(text):
    hits = pg.search_for(text); assert hits, text; r = hits[0]; return (r.x0 + r.x1) / 2, r
titles = [t for t in ("(a) 72-h back-trajectories, Jan–May", "(b) Upwind fire exposure by country")]
rects = []
for t in titles:
    hits = pg.search_for(t); assert len(hits) >= 1, f"title not found: {t}"; rects.append(hits[0]); pg.add_redact_annot(hits[0], fill=(1, 1, 1))
# panel centres from the tick labels: map axis spans 88–110° E (ticks 90 and 110), bar axis centred between 2023 and 2026
(xa90, _), (xa110, _) = centre_of("90"), centre_of("110")
cx_a = xa90 + (99 - 90) / (110 - 90) * (xa110 - xa90)
(x23, _), (x26, _) = centre_of("2023"), centre_of("2026")
cx_b = (x23 + x26) / 2
pg.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE)
fs = rects[0].height * 0.78
for (lab, cx), r in zip((("(a)", cx_a), ("(b)", cx_b)), rects):
    w = pymupdf.get_text_length(lab, fontsize=fs, fontname="helv")
    pg.insert_text((cx - w / 2, r.y1 - r.height * 0.22), lab, fontsize=fs, fontname="dejavu", fontfile=font, color=(0, 0, 0))
# map furniture for panel (a): lon/lat -> page points from the axis tick labels (x: 90 and 110 deg E, y: 10 and 28 deg N)
W_ = pg.get_text("words")
def tick(txt, left_of=None, below=None):
    c = [w for w in W_ if w[4] == txt and (left_of is None or w[2] < left_of) and (below is None or w[1] > below)]
    assert c, txt; return c
ylab10 = min(tick("10", left_of=xa90), key=lambda w: w[0]); ylab28 = min(tick("28", left_of=xa90), key=lambda w: w[0])
y10, y28 = (ylab10[1] + ylab10[3]) / 2, (ylab28[1] + ylab28[3]) / 2
X = lambda lon: xa90 + (lon - 90) / 20 * (xa110 - xa90)
Yp = lambda lat: y10 + (lat - 10) / 18 * (y28 - y10)
ink = (0.1, 0.1, 0.1); fsm = fs * 0.72
# north arrow at 108.3 deg E, 25.2-27.4 deg N
xn, yb, yt = X(108.3), Yp(25.2), Yp(27.4); hw = (X(108.9) - X(108.3))
pg.draw_polyline([(xn - hw, yb), (xn, yt), (xn + hw, yb), (xn, yb + (yt - yb) * 0.35), (xn - hw, yb)], color=ink, fill=ink, width=0.3)
wn = pymupdf.get_text_length("N", fontsize=fsm, fontname="helv")
pg.insert_text((xn - wn / 2, yt - fsm * 0.25), "N", fontsize=fsm, fontname="dejavu", fontfile=font, color=ink)
# 500 km scale bar along 11 deg N (1 deg lon = 111.32 cos(lat) km), starting at 104.0 deg E, on a white panel
lat_sb = 11.0; dlon = 500 / (111.32 * np.cos(np.radians(lat_sb))); x0, x1, ys = X(104.0), X(104.0 + dlon), Yp(lat_sb)
tk = (Yp(10) - Yp(10.25))
pg.draw_rect(pymupdf.Rect(x0 - 2 * fsm, ys - tk - 1.6 * fsm, x1 + 2.4 * fsm, ys + 2 * tk), color=None, fill=(1, 1, 1), fill_opacity=0.85)
pg.draw_line((x0, ys), (x1, ys), color=ink, width=0.9)
for xx in (x0, (x0 + x1) / 2, x1): pg.draw_line((xx, ys - tk), (xx, ys + tk), color=ink, width=0.6)
for txt, xx in (("0", x0), ("250", (x0 + x1) / 2), ("500 km", x1)):
    wt = pymupdf.get_text_length(txt, fontsize=fsm * 0.85, fontname="helv")
    pg.insert_text((xx - wt / 2, ys - tk - fsm * 0.25), txt, fontsize=fsm * 0.85, fontname="dejavu", fontfile=font, color=ink)
doc.save(f"{FD}/Figure_2.pdf")
dpi2 = dpi_for(pg.rect.width / 72)
pix = pg.get_pixmap(dpi=dpi2); pix.save(f"{FD}/Figure_2.png"); tif(f"{FD}/Figure_2.png", "Figure_2", dpi2)
print("Fig 2 panel centres (pt):", round(cx_a, 1), round(cx_b, 1), "page width", pg.rect.width)

# ---------- Fig 3 atmospheric context ----------
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7))
m = d.groupby(d.index.month).agg(pm=("pm25", "mean"))
a = ax[0]; a.bar(m.index, m.pm, color=B, width=0.7); a.axhline(37.5, color=MUT, ls="--", lw=0.8)
a.text(12.5, 39, "Thai 24-h standard", ha="right", fontsize=6.5, color=MUT); a.set_xticks(range(1, 13)); a.set_xticklabels(list("JFMAMJJASOND")); a.set_xlabel("Month")
a.set_ylabel("PM$_{2.5}$ (µg m$^{-3}$)"); label(a, "a"); grid(a)
a = ax[1]; c = d.dropna(subset=["ceres_clr", "e5_ssrdc", "aod_ceres"]); r = c.ceres_clr / c.e5_ssrdc
a.scatter(c.aod_ceres, r, s=3, color=B, alpha=0.3, lw=0, rasterized=True); xs = np.linspace(0, 1.8, 20)
a.plot(xs, 1.027 + 0.00122 * c.e5_tcwv.mean() - 0.230 * xs, color=Or, lw=1.6); a.axhline(1, color=MUT, ls="--", lw=0.8)
a.set_xlabel("CERES AOD$_{550}$"); a.set_ylabel("CERES / ERA5 clear-sky GHI"); label(a, "b"); grid(a)
a = ax[2]; s_ = d[(d.site_n_valid_ghi >= 80) & (d.e5_tcc_day < 30) & (d.aod_ceres < 0.4)]
rr = (s_.site_ghi_int_kWh / s_.e5_ssrd).dropna(); a.scatter(rr.index, rr, s=3, color=Aq, alpha=0.5, lw=0, rasterized=True)
t = (rr.index - rr.index[0]).days / 365.25; p = np.polyfit(t, rr, 1); a.plot(rr.index, np.polyval(p, t), color=Or, lw=1.6)
a.text(0.97, 0.95, f"{100 * p[0] / p[1]:.1f} % yr$^{{-1}}$".replace("-", "−", 1), transform=a.transAxes, ha="right", va="top", fontsize=7.5)
a.set_ylabel("On-site / ERA5 GHI"); label(a, "c"); a.set_ylim(0, 2)
a.xaxis.set_major_locator(mdates.YearLocator()); a.xaxis.set_major_formatter(mdates.DateFormatter("%Y")); grid(a)
fig.tight_layout(); save(fig, "Figure_3")

# ---------- Fig 4 PV response ----------
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.8))
pr = pd.read_csv(f"{S5}/R6_fig4a_partial_hinge.csv", index_col=0, parse_dates=True); Hh = R["R6_fig4a_hinge"]
a = ax[0]; a.scatter(pr.aod, pr.rel, s=3, color=B, alpha=0.2, lw=0, rasterized=True)
bm = pr.groupby(pd.cut(pr.aod, np.arange(0, 1.36, 0.15)), observed=True).rel.agg(["mean", "sem", "count"]); bm = bm[bm["count"] >= 10]
a.errorbar([i.mid for i in bm.index], bm["mean"], yerr=1.96 * bm["sem"], fmt="o", color=INK, ms=3.5, capsize=2, lw=0.9, label="Binned mean, 95 % CI")
xs = np.linspace(0, 1.3, 60); lin = np.polyfit(pr.aod, pr.rel, 1)
a.plot(xs, np.polyval(lin, xs), color=Or, lw=1.3, ls="--", label="Linear"); a.plot(xs, 1 + Hh["b"] * np.clip(xs - Hh["knot"], 0, None), color=Or, lw=1.6, label=f"Hinge at AOD {Hh['knot']:.2f}")
a.set_xlim(0, 1.4); a.set_ylim(0.5, 1.3); a.set_xlabel("CERES AOD$_{550}$"); a.set_ylabel("Relative clear-day PV yield"); a.legend(fontsize=6, frameon=False, loc="lower left")
label(a, "a"); grid(a)
# (b) out-of-sample skill of the soiling metrics: RMSE reduction vs the model without a soiling term (rev25)
A_ = pd.read_csv(f"{ROB}/A_soiling_metric_comparison.csv"); A_ = A_[~A_.season_harmonic].set_index("metric")
SHORT = [("PM2.5 dose, partial rain cleaning (primary)", "PM$_{2.5}$, partial (selected)", Or), ("PM2.5 dose, 2 mm reset (rev24 primary)", "PM$_{2.5}$, reset 2 mm", B),
         ("PM2.5 dose, 0.5 mm reset", "PM$_{2.5}$, reset 0.5 mm", B), ("PM2.5 dose, 5 mm reset", "PM$_{2.5}$, reset 5 mm", B),
         ("Coarse dose, 2 mm reset", "Coarse, reset 2 mm", GR), ("Coarse dose, 0.5 mm reset", "Coarse, reset 0.5 mm", GR),
         ("HSU-weighted dose, 2 mm reset", "HSU-type, reset 2 mm", GR), ("HSU-weighted dose, 0.5 mm reset", "HSU-type, reset 0.5 mm", GR),
         ("Days since reset, 20 mm", "Days, reset 20 mm", Aq)]
a = ax[1]; yy = np.arange(len(SHORT))[::-1]
a.barh(yy, [A_.loc[k, "loyo_gain_vs_no_soil_pct"] for k, _, _ in SHORT], height=0.62, color=[c for _, _, c in SHORT])
a.scatter([A_.loc[k, "lomo_gain_vs_no_soil_pct"] for k, _, _ in SHORT], yy, s=9, color=INK, zorder=3, label="Month blocks held out")
a.axvline(0, color=MUT, lw=0.8); a.set_yticks(yy); a.set_yticklabels([s for _, s, _ in SHORT], fontsize=5.6)
a.set_xlabel("Out-of-sample RMSE reduction (%)")   # bars: years held out; dots: months held out (explained in the caption)
label(a, "b"); a.grid(axis="x", color="#e6e5e0", lw=0.6); a.set_axisbelow(True)
# (c) modelled soiling loss, January-June 2024, partial-cleaning dose
ds = pd.read_csv(f"{S5}/dose_series_rev25.csv", parse_dates=["date"]).set_index("date")
a = ax[2]; seg = ds.loc["2024-01-01":"2024-06-30"]; y_ = seg.loss_pct
a.fill_between(seg.index, 0, y_, color=V, alpha=0.35, lw=0); a.plot(seg.index, y_, color=V, lw=1)
for cd in pd.to_datetime(["2024-03-09", "2024-06-23"]): a.axvline(cd, color=INK, lw=0.8, ls="--")
rain = seg.index[seg.tp >= 2]; a.plot(rain, np.full(len(rain), -0.4), "|", color=B, ms=5)
lt = seg.index[(seg.tp >= 0.1) & (seg.tp < 2)]; a.plot(lt, np.full(len(lt), -0.4), "|", color=B, ms=2.5, alpha=0.6)
a.set_ylabel("Modelled soiling loss (%)"); label(a, "c")
a.xaxis.set_major_locator(mdates.MonthLocator()); a.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
pass   # washes (dashed lines) and rain ticks are explained in the caption
a.tick_params(axis="x", labelsize=6.5)
grid(a); fig.tight_layout(); save(fig, "Figure_4")

# ---------- Fig 5 attribution ----------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
x = np.arange(4); w = 0.38
S = R["R3_shapley"]; ex = json.load(open(f"{ROB}/../checks_out/exposure_common.json"))["p850"]["loss_days"]   # same days as the losses (Table 4)
imp = S["p850"]["shapley_share"]; ci = json.load(open(f"{ROB}/../checks_out/checks_rev25b.json"))["A_CI_total_share"]   # CI of the total loss share (Table 4)
a = ax[0]; a.bar(x - w / 2, [ex[c] for c in C], w, color=B, label="Fire exposure")
a.bar(x + w / 2, [imp[c] for c in C], w, color=Or, label="Fire-attributable PV loss", yerr=np.array([[imp[c] - ci[c][0] for c in C], [ci[c][1] - imp[c] for c in C]]), capsize=2, error_kw=dict(lw=0.8))
a.axhline(0, color=MUT, lw=0.6); a.set_xticks(x); a.set_xticklabels(labs); a.set_ylabel("Share (%)"); a.set_ylim(-35, 105); a.legend(fontsize=6.5, frameon=False, loc="upper right")
label(a, "a"); grid(a)
a = ax[1]; st = {"850 hPa": S["p850"]["shapley_share"], "700 hPa": S["p700"]["shapley_share"], "3-D from 850 hPa": S["3d850"]["shapley_share"]}; order = ["3-D from 850 hPa", "700 hPa", "850 hPa"]; bot = np.zeros(3)
for c, col, lb in zip(C, (Or, B, Aq, GR), labs):
    v = np.array([max(st[k][c], 0) for k in order]); a.barh(order, v, left=bot, color=col, label=lb, edgecolor="white", lw=1); bot += v
for i, k in enumerate(order):
    a.text(st[k]["THA"] / 2, i, f"{st[k]['THA']:.0f}%", ha="center", va="center", color="white", fontsize=7)
    if st[k]["MMR"] > 10: a.text(st[k]["THA"] + st[k]["MMR"] / 2, i, f"{st[k]['MMR']:.0f}%", ha="center", va="center", color="white", fontsize=7)
a.set_xlim(0, 100); a.set_xlabel("Share of fire-attributable PV loss (%)"); a.legend(fontsize=6.5, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.45, -0.22))
label(a, "b"); fig.tight_layout(); save(fig, "Figure_5")

# ---------- Fig 6 losses + washing economics ----------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
f6 = pd.read_csv(f"{S5}/R3_fig6a_components.csv", index_col=0); a = ax[0]; bot = np.zeros(len(f6))
for col_, col, lab in (("thai_atm_MWh", Or, "Atmospheric: Thai fires"), ("abroad_atm_MWh", Y, "Atmospheric: fires abroad"), ("nonfire_atm_MWh", B, "Atmospheric: non-fire aerosol"), ("dep_JanMay", V, "Deposition (soiling)")):
    a.bar(f6.index.astype(str), f6[col_].values, bottom=bot, color=col, width=0.62, label=lab, edgecolor="white", lw=1); bot += f6[col_].values
for i, t_ in enumerate(bot): a.text(i, t_ + 2, f"{t_:.0f}", ha="center", fontsize=7)
a.set_ylabel("PV energy loss, Jan–May (MWh)"); a.set_ylim(0, 215); a.set_yticks([0, 50, 100, 150]); a.legend(fontsize=5.6, frameon=False, ncol=2, loc="upper right", columnspacing=0.8, handlelength=1.2)
a.set_xticks(range(len(f6))); a.set_xticklabels(["2023*", "2024", "2025", "2026"]); label(a, "a"); grid(a)
a = ax[1]; cs_ = pd.read_csv(f"{S4}/T14c_cleaning_cost_sensitivity.csv"); s2 = cs_[cs_.rain_reset_mm == 2.0]; s5 = cs_[cs_.rain_reset_mm == 5.0]
a.plot(s2.wash_cost_THB / 1000, s2.saving_vs_none_THB / 1000, "o-", color=Or, ms=4, lw=1.4, label="Rain cleans fully at 2 mm (selected)")
a.plot(s5.wash_cost_THB / 1000, s5.saving_vs_none_THB / 1000, "s--", color=B, ms=4, lw=1.2, label="Rain cleans fully at 5 mm")
prevD = None
for _, r_ in s2[s2.saving_vs_none_THB > 0].iterrows():
    if r_.D_star == prevD: continue
    prevD = r_.D_star; a.annotate(f"D*={r_.D_star:g}", (r_.wash_cost_THB / 1000, r_.saving_vs_none_THB / 1000), xytext=(5, 4), textcoords="offset points", fontsize=6, color=Or)
be = R["R2_breakeven"]
ytop = 1.15 * max(s2.saving_vs_none_THB.max(), s5.saving_vs_none_THB.max()) / 1000
for k, col in (("2mm", Or), ("5mm", B)):
    c_ = be[k]["breakeven_wash_cost_THB"] / 1000; a.axvline(c_, color=col, lw=0.8, ls=":")
    a.text(c_ * 0.97, ytop * (0.55 if k == "2mm" else 0.72), f"break-even\n≈{c_:.0f} kTHB", fontsize=6, color=col, va="center", ha="right")
a.axhline(0, color=MUT, lw=0.8); a.set_xscale("log"); a.set_xticks([5, 10, 20, 50, 100]); a.set_xticklabels(["5", "10", "20", "50", "100"])
a.set_xlabel("Cost per fleet wash (kTHB)"); a.set_ylabel("Saving vs no washing, 3.5 yr (kTHB)"); a.legend(fontsize=6.2, frameon=False, loc="upper right"); a.set_ylim(-0.04 * ytop, ytop)
label(a, "b"); grid(a); fig.tight_layout(); save(fig, "Figure_6")

# ---------- graphical abstract (unchanged design, 1000 dpi) ----------
J = json.load(open(f"{S7}/joint_bootstrap_total_loss.json"))
fig = plt.figure(figsize=(5.12, 2.6)); g = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.05, 0.9], wspace=0.25)
a = fig.add_subplot(g[0]); a.barh(["Other", "Laos", "Myanmar", "Thailand"], [ex[c] for c in ["OTH", "LAO", "MMR", "THA"]], color=[GR, Aq, B, Or])
a.set_xlim(0, 100); a.set_title("Upwind fire exposure", fontsize=8); a.set_xlabel("%", fontsize=7); a.tick_params(labelsize=7)
for i, c in enumerate(["OTH", "LAO", "MMR", "THA"]): a.text(ex[c] + 2, i, f"{ex[c]:.0f}", va="center", fontsize=7)
b = fig.add_subplot(g[1]); b.axis("off"); b.set_xlim(0, 1); b.set_ylim(0, 1)
b.text(0.5, 0.88, "2.51 MWp hospital\nrooftop PV", ha="center", va="center", fontsize=7.5, fontweight="bold")
b.text(0.5, 0.62, f"Haze loss\n{J['point_hinge']['tot']:.0f}–{J['point_linear']['tot']:.0f} MWh yr$^{{-1}}$\n(95 % CI {J['CI95_pooled_tot'][0]:.0f}–{J['CI95_pooled_tot'][1]:.0f})", ha="center", va="center", fontsize=7)
b.annotate("", xy=(0.98, 0.38), xytext=(0.02, 0.38), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", color=MUT, lw=1.2))
b.text(0.5, 0.16, f"≈{100 * J['point_linear']['atm'] / J['point_linear']['tot']:.0f} % atmospheric,\nlight rain cleans partly", ha="center", va="center", fontsize=6.5, color=MUT)
c3 = fig.add_subplot(g[2]); c3.barh(["Other", "Laos", "Myanmar", "Thailand"], [imp[c] for c in ["OTH", "LAO", "MMR", "THA"]], color=[GR, Aq, B, Or])
c3.set_xlim(0, 100); c3.set_title("Fire-attributable PV loss", fontsize=8); c3.set_xlabel("%", fontsize=7); c3.tick_params(labelsize=7); c3.set_yticklabels([])
for i, c in enumerate(["OTH", "LAO", "MMR", "THA"]): c3.text(imp[c] + 2, i, f"{imp[c]:.0f}", va="center", fontsize=7)
fig.suptitle("Upwind fire exposure versus attributed PV loss", fontsize=9.5, fontweight="bold", y=1.0)
fig.savefig(f"{FD}/Graphical_abstract.pdf", bbox_inches="tight"); fig.savefig(f"{FD}/Graphical_abstract.png", dpi=DPI, bbox_inches="tight"); plt.close(fig)
tif(f"{FD}/Graphical_abstract.png", "Graphical_abstract")
for n in ["Figure_1", "Figure_2", "Figure_3", "Figure_4", "Figure_5", "Figure_6", "Graphical_abstract"]:
    im = Image.open(f"{FD}/{n}.tif"); print(n, im.size, im.info.get("dpi"))
print("ok")
