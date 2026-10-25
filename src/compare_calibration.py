"""BEACON: before vs after grid calibration, key numbers side by side (prints + one figure).
Before: real_grid_sim.py / chain_sim.py (Task 3 grid settings, 10 % loss, battery sized in pu of demand).
After:  real_grid_cal.py (calibrated to Grid-India, 4,500 MW loss, battery fixed in MW)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "outputs", ROOT / "figures"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
pd.set_option("display.width", 250)

old = pd.read_csv(OUT / "real_grid_summary.csv")
new = pd.read_csv(OUT / "real_grid_cal_summary.csv")
old = old[(old.event == 0.10)].copy()
new = new[(new.event == "4500MW")].copy()
for df in (old, new):
    df["year"] = df.year.astype(str)

rows = []
for d in [0.8, 1.0, 1.2]:
    for c in ["quadratic", "linear", "lit_+100%"]:
        for s in [0.9, 0.82, 0.7]:
            for y in ["2021", "2022", "2023", "2024", "2025", "all"]:
                o = old[(old.design_demand == d) & (old.curve == c) & (old.soh == s) & (old.year == y)]
                n = new[(new.design_demand == d) & (new.curve == c) & (new.soh == s) & (new.year == y)]
                if len(o) and len(n):
                    rows.append(dict(design_demand=d, curve=c, soh=s, year=y,
                                     before_pct=o.pct_hours_aging_matters.iloc[0],
                                     after_pct=n.pct_hours_aging_matters.iloc[0],
                                     before_drocof_max=o.drocof_max.iloc[0], after_drocof_max=n.drocof_max.iloc[0],
                                     before_dnadir_min=o.dnadir_min.iloc[0], after_dnadir_min=n.dnadir_min.iloc[0]))
T = pd.DataFrame(rows)
T.to_csv(OUT / "calibration_before_after.csv", index=False)
print("% of hours where aging matters, before vs after calibration (quadratic):")
q = T[(T.curve == "quadratic")]
print(q.pivot_table(index=["design_demand", "soh"], columns="year", values=["before_pct", "after_pct"]).round(2).to_string())
print("\nall hours, largest RoCoF increase and nadir decrease vs fresh battery:")
print(T[T.year == "all"][["design_demand", "curve", "soh", "before_drocof_max", "after_drocof_max",
                          "before_dnadir_min", "after_dnadir_min"]].round(4).to_string(index=False))

fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
yrs = ["2021", "2022", "2023", "2024", "2025"]
for a, d in zip(ax, [0.8, 1.0, 1.2]):
    for s, col in [(0.9, "tab:blue"), (0.82, "tab:orange")]:
        x = q[(q.design_demand == d) & (q.soh == s)].set_index("year").reindex(yrs)
        a.plot(yrs, x.before_pct, "--", color=col, marker="o", ms=3, label=f"SoH {s}, before (textbook grid, 10 %)")
        a.plot(yrs, x.after_pct, "-", color=col, marker="s", ms=4, label=f"SoH {s}, after (calibrated, 4,500 MW)")
    a.set(title=f"battery sized at demand ratio {d}", xlabel="year")
ax[0].set_ylabel("% of hours where aging matters")
ax[0].legend(fontsize=7)
fig.suptitle("Before vs after grid calibration, quadratic R(SoH)")
fig.tight_layout()
fig.savefig(FIG / "calibration_before_after.png", dpi=150)
print("saved outputs/calibration_before_after.csv, figures/calibration_before_after.png")
