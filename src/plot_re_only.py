"""BEACON: figures for the solar-only / wind-only study (re_only.py outputs)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import power_sim as ps

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "outputs", ROOT / "figures"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
COL = {"LFP": "tab:blue", "NMC": "tab:orange", "NCA": "tab:green"}
LS = {"solar": "-", "wind": "--"}

disp = pd.read_csv(OUT / "re_dispatch_fresh.csv", parse_dates=["t"])
S = pd.read_csv(OUT / "re_soh.csv")
R = pd.read_csv(OUT / "re_events.csv")
H = pd.read_csv(OUT / "re_events_by_hour.csv")

# 1. average day: plant output, battery dispatch, state of charge
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
hod = disp.t.dt.hour
for src, c in [("solar", "tab:orange"), ("wind", "tab:blue")]:
    ax[0].plot(disp.groupby(hod)[f"{src}_cf"].mean() * 100, color=c, label=f"{src} plant output (MW)")
    ax[0].plot(disp.groupby(hod)[f"{src}_p"].mean(), color=c, ls="--", label=f"{src} battery dispatch (MW, + = discharge)")
    ax[1].plot(disp.groupby(hod)[f"{src}_soc"].mean() * 100, color=c, label=src)
ax[0].axhline(0, color="k", lw=0.8)
ax[0].set(xlabel="hour of day (IST)", ylabel="MW (100 MW plant, 25 MW battery)",
          title="Average day, Kutch 2025 (fresh battery)")
ax[1].axhspan(10, 30, color="grey", alpha=0.15, label="charge reserved for frequency support")
ax[1].set(xlabel="hour of day (IST)", ylabel="battery state of charge (%)", title="When does the battery have spare charge?")
for a in ax:
    a.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "re_average_day.png", dpi=150)
plt.close(fig)

# 2. SoH and power ceiling per year
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
for c in COL:
    for src in ["solar", "wind"]:
        x = S[(S.source == src) & (S.chem == c) & (S.temp_case == "site")]
        ax[0].plot(x.year, x.SoH, LS[src], color=COL[c], marker="o", ms=3, label=f"{c}, {src}")
        b = x[x.beyond_test]
        ax[0].plot(b.year, b.SoH, "o", color="white", mec=COL[c], ms=6)
        pc = 25 / np.array([float(ps.r_curve("quadratic", s)) for s in x.SoH])
        ax[1].plot(x.year, pc, LS[src], color=COL[c], marker="o", ms=3, label=f"{c}, {src}")
ax[1].axhline(22.5, color="k", ls=":", lw=1, label="reserve 12.5 MW + 10 MW design trip")
ax[0].set(xlabel="years of operation", ylabel="SoH (Sandia Model 1)",
          title="Battery health (open marker: beyond Sandia test length)")
ax[1].set(xlabel="years of operation", ylabel="power ceiling (MW, quadratic R(SoH))", title="Battery power ceiling")
ax[0].legend(fontsize=7, ncol=2)
ax[1].legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(FIG / "re_soh_power_years.png", dpi=150)
plt.close(fig)

# 3. % of event hours not covered, design event, unaware vs aware; and stress event load shedding
base = (R.temp_case == "site") & (R.curve == "quadratic")
fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
for c in COL:
    for src in ["solar", "wind"]:
        x = R[base & (R.event == "10% trip (design)") & (R.operation == "SoH-unaware") & (R.source == src) & (R.chem == c)]
        ax[0].plot(x.year, x.pct_not_covered, LS[src], color=COL[c], marker="o", ms=3, label=f"{c}, {src}")
        x = R[base & (R.event == "10% trip (design)") & (R.operation == "SoH-aware") & (R.source == src) & (R.chem == c)]
        ax[1].plot(x.year, x.pct_not_covered, LS[src], color=COL[c], marker="o", ms=3, label=f"{c}, {src}")
        x = R[base & (R.event == "25% trip (stress)") & (R.operation == "SoH-unaware") & (R.source == src) & (R.chem == c)]
        ax[2].plot(x.year, x.pct_load_shedding, LS[src], color=COL[c], marker="o", ms=3, label=f"{c}, {src}")
ax[0].set(title="Design event (10 MW trip), SoH-UNAWARE operation", ylabel="% of event hours not covered")
ax[1].set(title="Design event (10 MW trip), SoH-AWARE operation", ylabel="% of event hours not covered",
          ylim=ax[0].get_ylim())
ax[2].set(title="Stress event (25 MW trip), SoH-unaware", ylabel="% of event hours with load shedding (> 0.6 Hz)")
for a in ax:
    a.set_xlabel("years of operation")
    a.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(FIG / "re_events_years.png", dpi=150)
plt.close(fig)

# 4. when does it fail? design event, year 10, SoH-unaware
fig, ax = plt.subplots(figsize=(9, 4.5))
for c in ["NMC", "NCA"]:
    for src in ["solar", "wind"]:
        x = H[(H.source == src) & (H.chem == c) & (H.year == 10) & (H.operation == "SoH-unaware")
              & (H.event == "10% trip (design)")]
        ax.plot(x.hour, x.pct_not_covered, LS[src], color=COL[c], marker="o", ms=3, label=f"{c}, {src}")
ax.set(xlabel="hour of day (IST)", ylabel="% of that hour's events not covered",
       title="Year 10, design event, SoH-unaware: when the aged battery falls short (LFP: never)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "re_fail_by_hour.png", dpi=150)
plt.close(fig)
print("saved re_average_day.png, re_soh_power_years.png, re_events_years.png, re_fail_by_hour.png")
