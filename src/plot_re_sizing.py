"""BEACON: figures for the battery and reserve sizing sweep (re_sizing.py output)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "outputs", ROOT / "figures"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
COL = {"LFP": "tab:blue", "NMC": "tab:orange", "NCA": "tab:green"}
LS = {10.0: ":", 12.5: "-", 15.0: "--"}

R = pd.read_csv(OUT / "re_sizing.csv")
y10 = R[R.year == 10]

# 1. design event, year 10, SoH-unaware: % not covered vs battery duration, per power and source
fig, ax = plt.subplots(2, 3, figsize=(16, 8), sharey="row")
for i, src in enumerate(["solar", "wind"]):
    for j, mw in enumerate([15.0, 25.0, 35.0]):
        a = ax[i, j]
        for c in COL:
            for res in [10.0, 12.5, 15.0]:
                x = y10[(y10.source == src) & (y10.bat_MW == mw) & (y10.reserve_MW == res) & (y10.chem == c)
                        & (y10.operation == "SoH-unaware") & (y10.event == "10% trip (design)")]
                if len(x):
                    a.plot(x.hours, x.pct_not_covered, LS[res], color=COL[c], marker="o", ms=4,
                           label=f"{c}, reserve {res:g} MW")
        a.set(title=f"{src}, battery {mw:g} MW", xticks=[1, 2, 4])
        if i == 1:
            a.set_xlabel("battery duration (h), energy = power x duration")
        if j == 0:
            a.set_ylabel("% of event hours not covered")
ax[0, 2].legend(fontsize=7, ncol=1)
fig.suptitle("Design event (10 MW trip), year 10, SoH-unaware operation. SoH-aware operation: 0 % for every size")
fig.tight_layout()
fig.savefig(FIG / "re_sizing_design.png", dpi=150)
plt.close(fig)

# 2. cost of SoH-aware operation and stress-event load shedding, reserve 12.5 MW
d = y10[(y10.event == "10% trip (design)") & (y10.reserve_MW == 12.5)]
t = d.pivot_table(index=["bat_MW", "hours", "source", "chem"], columns="operation", values="energy_shifted_MWh")
t["cost"] = 100 * (1 - t["SoH-aware"] / t["SoH-unaware"])
t = t.reset_index()
s = y10[(y10.event == "25% trip (stress)") & (y10.reserve_MW == 12.5) & (y10.operation == "SoH-unaware")]
fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
for c in COL:
    for src, mk in [("solar", "o"), ("wind", "s")]:
        for hrs, ls in [(1.0, ":"), (2.0, "-"), (4.0, "--")]:
            x = t[(t.chem == c) & (t.source == src) & (t.hours == hrs)]
            ax[0].plot(x.bat_MW, x.cost, ls, marker=mk, color=COL[c], ms=4,
                       label=f"{c}, {src}, {hrs:g} h" if src == "solar" or hrs == 2.0 else None)
        x = s[(s.chem == c) & (s.source == src) & (s.hours == 2.0)]
        ax[1].plot(x.bat_MW, x.pct_load_shedding, "-" if src == "solar" else "--", marker=mk, color=COL[c],
                   label=f"{c}, {src}")
ax[0].set(xlabel="battery power (MW)", ylabel="% less energy shifted than SoH-unaware",
          title="Year 10: price of SoH-aware operation (reserve 12.5 MW)", xticks=[15, 25, 35])
ax[1].set(xlabel="battery power (MW)", ylabel="% of event hours with load shedding",
          title="Year 10: stress event (25 MW trip), 2 h battery, SoH-unaware", xticks=[15, 25, 35])
ax[0].legend(fontsize=6, ncol=3)
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "re_sizing_cost_stress.png", dpi=150)
plt.close(fig)
print("saved re_sizing_design.png, re_sizing_cost_stress.png")
