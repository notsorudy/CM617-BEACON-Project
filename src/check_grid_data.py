"""BEACON: quality checks on the hourly Grid-India table (outputs/grid_india_hourly.csv).
Prints checks and saves figures/grid_india_data_check.png."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "outputs", ROOT / "figures"
pd.set_option("display.width", 250)

h = pd.read_csv(OUT / "grid_india_hourly.csv", parse_dates=["time"])
full = pd.date_range(h.time.min(), h.time.max(), freq="h")
print(f"hours {len(h)}, expected {len(full)} from {full[0]} to {full[-1]}, missing hours {len(full.difference(h.time))}")
h["period"] = np.where(h.time < "2022-10-01", "A 5-min 2021-09..2022-09",
                       np.where(h.time < "2024-01-01", "B 10-s 2022-10..2023-12", "C hourly 2024-01..2025-06"))

cols = ["demand_MW", "thermal_MW", "hydro_MW", "gas_MW", "nuclear_MW", "wind_MW", "solar_MW", "total_gen_MW",
        "tag_INTRA_SYSTEM_INTERTIA", "tag_ER_Total_Inertia", "tag_SYS_IN"]
print("\nper period: min / median / max, NaN count, values <= 0")
for p, g in h.groupby("period"):
    print(f"--- {p} ({len(g)} hours)")
    for c in cols:
        v = g[c]
        if v.notna().sum() == 0:
            print(f"  {c:26s} not in this period")
            continue
        print(f"  {c:26s} {v.min():11.1f} {v.median():11.1f} {v.max():11.1f}   NaN {int(v.isna().sum()):5d}   <=0 {int((v <= 0).sum()):5d}")

# time zone: solar peak hour, and night solar
h["hod"] = h.time.dt.hour
for p, g in h.groupby("period"):
    prof = g.groupby("hod").solar_MW.mean()
    print(f"{p}: solar mean peaks at hour {prof.idxmax()} (file time); mean solar at 00-04 h: {prof.loc[0:4].mean():.1f} MW, "
          f"at peak: {prof.max():.0f} MW")

# demand vs sum of listed sources (periods A and B)
ab = h[h.period != "C hourly 2024-01..2025-06"].copy()
src = ["thermal_MW", "hydro_MW", "gas_MW", "nuclear_MW", "wind_MW"]
ab["sum_sources"] = ab[src].sum(axis=1) + ab.solar_MW.clip(lower=0)
r = ab.sum_sources / ab.demand_MW
print(f"\nperiods A+B: (sum of listed sources) / demand: median {r.median():.3f}, 1st pct {r.quantile(.01):.3f}, "
      f"99th pct {r.quantile(.99):.3f}")
c = h[h.period == "C hourly 2024-01..2025-06"]
print(f"period C: total_gen / demand: median {(c.total_gen_MW / c.demand_MW).median():.3f}")

# continuity at the boundaries (same quantity?)
for a, b in [("2022-09-24", "2022-10-07"), ("2023-12-25", "2024-01-07")]:
    w = h[(h.time >= a) & (h.time < b)].set_index("time")
    d = w.resample("D")[["demand_MW", "wind_MW", "solar_MW"]].mean().round(0)
    print(f"\ndaily means around the format change {a} .. {b}:")
    print(d.to_string())

# renewable share
h["vre_share"] = (h.wind_MW.clip(lower=0) + h.solar_MW.clip(lower=0)) / h.demand_MW
h["year"] = h.time.dt.year
print("\nhourly (wind + solar) / demand by year:")
print(h.groupby("year").vre_share.describe(percentiles=[.5, .95]).round(3).to_string())
print(f"max share {h.vre_share.max():.3f} at {h.loc[h.vre_share.idxmax(), 'time']}")

# inertia tags
for t in ["tag_INTRA_SYSTEM_INTERTIA", "tag_ER_Total_Inertia", "tag_SYS_IN"]:
    nz = h[h[t] > 0]
    if len(nz):
        print(f"\n{t}: non-zero in {len(nz)} hours, from {nz.time.min()} to {nz.time.max()}; "
              f"min {nz[t].min():.3f} median {nz[t].median():.3f} max {nz[t].max():.3f}; "
              f"hours with exactly repeated value vs previous hour: {int((nz[t].diff() == 0).sum())}")
nz = h[(h.tag_SYS_IN > 0) & h.thermal_MW.notna()].copy()
nz["sync_share"] = (nz.thermal_MW + nz.hydro_MW + nz.gas_MW + nz.nuclear_MW) / nz.demand_MW
print(f"corr(tag_SYS_IN, sync generation MW) = {nz.tag_SYS_IN.corr(nz.thermal_MW + nz.hydro_MW + nz.gas_MW + nz.nuclear_MW):.3f}; "
      f"corr(tag_SYS_IN, sync share) = {nz.tag_SYS_IN.corr(nz.sync_share):.3f}; "
      f"corr(tag_SYS_IN, vre share) = {nz.tag_SYS_IN.corr(nz.vre_share):.3f}")

# figure
fig, ax = plt.subplots(2, 2, figsize=(15, 9))
m = h.set_index("time").resample("W").mean(numeric_only=True)
for c_, lab in [("demand_MW", "demand"), ("thermal_MW", "thermal (coal)"), ("hydro_MW", "hydro"),
                ("wind_MW", "wind"), ("solar_MW", "solar")]:
    ax[0, 0].plot(m.index, m[c_] / 1000, label=lab)
ax[0, 0].set(ylabel="GW (weekly mean)", title="All-India, Grid-India SCADA (Mendeley 10.17632/y58jknpgs8.2)")
ax[0, 0].legend(fontsize=8)
for y, g in h.groupby("year"):
    ax[0, 1].plot(g.groupby("hod").vre_share.mean(), label=str(y))
ax[0, 1].set(xlabel="hour of day (file time)", ylabel="(wind + solar) / demand", title="Average day, by year")
ax[0, 1].legend()
ax[1, 0].plot(m.index, m.vre_share, color="tab:green")
ax[1, 0].set(ylabel="(wind + solar) / demand, weekly mean", title="Variable renewable share over time")
for t in ["tag_INTRA_SYSTEM_INTERTIA", "tag_ER_Total_Inertia", "tag_SYS_IN"]:
    s = h.set_index("time")[t]
    ax[1, 1].plot(s.index, s.where(s > 0), lw=0.5, label=t)
ax[1, 1].set(ylabel="value as stored (unit not given)", title="Inertia-labelled tags (hourly mean, zeros hidden)")
ax[1, 1].legend(fontsize=8)
for a in ax.flat:
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "grid_india_data_check.png", dpi=150)
print("\nsaved figures/grid_india_data_check.png")
