"""BEACON: compact figures for the 2-page report (column width), from the main-study outputs.
fig_A: SoH and power ceiling per year (re_soh.csv); fig_B: frequency after the design trip at two real hours
(re_freq_traces.csv); fig_C: % of trip hours not covered per year, SoH-unaware (re_events.csv)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT, REP = ROOT / "outputs", ROOT / "report"
REP.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 7, "axes.grid": True, "grid.alpha": 0.3, "lines.linewidth": 1.1,
                     "legend.fontsize": 6, "axes.titlesize": 7})
COL = {"LFP": "tab:blue", "NMC": "tab:orange", "NCA": "tab:green"}
W = 3.45                                   # IEEE column width, inches

a, b, c = json.load(open(OUT / "resistance_curve_coefs.json"))["quadratic_a_b_c"]
r = lambda s: (a + b * (1 - s) + c * (1 - s) ** 2) / a

# A: SoH and power ceiling (solar duty; wind within 0.01 SoH)
S = pd.read_csv(OUT / "re_soh.csv")
S = S[(S.temp_case == "site") & (S.source == "solar")]
fig, ax = plt.subplots(1, 2, figsize=(W, 1.55))
for ch in COL:
    x = S[S.chem == ch]
    ax[0].plot(x.year, x.SoH, color=COL[ch], marker="o", ms=1.8, label=ch)
    bt = x[x.beyond_test]
    ax[0].plot(bt.year, bt.SoH, "o", ms=3, mfc="white", mec=COL[ch])
    ax[1].plot(x.year, 25 / r(x.SoH), color=COL[ch], marker="o", ms=1.8, label=ch)
ax[1].axhline(22.5, color="k", ls=":", lw=0.8)
ax[1].text(15, 22.6, "needed: 12.5 + 10 MW", ha="right", va="bottom", fontsize=5.5)
ax[0].set(xlabel="year", ylabel="SoH", title="(a) state of health")
ax[1].set(xlabel="year", ylabel="MW", title="(b) power ceiling")
ax[0].legend(loc="center left", bbox_to_anchor=(0.3, 0.42))
fig.tight_layout(pad=0.3)
fig.savefig(REP / "fig_A.png", dpi=300)
plt.close(fig)

# B: frequency after the 10 MW trip at the two example hours
T = pd.read_csv(OUT / "re_freq_traces.csv")
E = pd.read_csv(OUT / "re_freq_example_hours.csv")
T, E = T[T.event == "10% trip (design)"], E[E.event == "10% trip (design)"]
fig, ax = plt.subplots(1, 2, figsize=(W, 1.8), sharey=True)
for j, src in enumerate(["solar", "wind"]):
    for ch, mode, col, ls, lab in [("fresh", "SoH-unaware", "k", "-", "fresh / LFP / NCA aware"),
                                   ("NMC", "SoH-unaware", COL["NMC"], "-", "NMC unaware"),
                                   ("NCA", "SoH-unaware", COL["NCA"], "-", "NCA unaware")]:
        x = T[(T.source == src) & (T.chem == ch) & (T.operation == mode)]
        ax[j].plot(x.t, x.f_Hz, ls, color=col, label=lab)
    ax[j].axhline(49.4, color="red", ls="--", lw=0.7)
    hr = E[E.source == src].hour.iloc[0][5:16]
    ax[j].set(xlabel="time after trip (s)", xlim=(0, 10), title=f"({'ab'[j]}) {src}, {hr}")
ax[0].set_ylabel("frequency (Hz)")
ax[0].text(9.8, 49.42, "49.4 Hz load shedding", color="red", ha="right", va="bottom", fontsize=5.5)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False)
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.9))
fig.savefig(REP / "fig_B.png", dpi=300)
plt.close(fig)

# C: % of trip hours not covered per year, SoH-unaware (SoH-aware: 0 everywhere)
R = pd.read_csv(OUT / "re_events.csv")
R = R[(R.temp_case == "site") & (R.curve == "quadratic") & (R.event == "10% trip (design)")
      & (R.operation == "SoH-unaware")]
fig, ax = plt.subplots(figsize=(W, 1.55))
for ch in COL:
    for src, ls in [("solar", "-"), ("wind", "--")]:
        x = R[(R.chem == ch) & (R.source == src)]
        ax.plot(x.year, x.pct_not_covered, ls, color=COL[ch], marker="o", ms=1.8, label=f"{ch}, {src}")
ax.text(15, 0.85, "SoH-aware operation: 0 % in every case", fontsize=6, ha="right", va="center")
ax.set(xlabel="year", ylabel="% of trip hours\nnot covered", ylim=(-0.05, 2.0))
ax.legend(ncol=3, loc="upper left")
fig.tight_layout(pad=0.3)
fig.savefig(REP / "fig_C.png", dpi=300)
plt.close(fig)
print("saved report/fig_A.png, fig_B.png, fig_C.png")
