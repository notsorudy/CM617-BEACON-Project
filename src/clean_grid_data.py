"""
BEACON: clean the hourly Grid-India table and compute the hourly renewable and synchronous shares.

An hour is dropped if any rule fires:
  zero        demand <= 0 (SCADA outage)
  dropout     demand more than 15% away from its 5-hour rolling median (part of the hour missing);
              blank (zero) hours are left out of the median so they cannot hide a partial hour
  low_demand  demand below 60% of that month's median demand (implausible for all-India)
  balance     sum of sources / demand outside 0.95 .. 1.08 (2021-09 to 2023-12, sources listed)
              total generation / demand outside 0.95 .. 1.08 (2024-01 to 2025-06)
  stuck       demand, wind, total generation, or daytime solar (> 500 MW) exactly equal for 3 or
              more hours in a row (frozen SCADA value)
  night_solar solar above 1000 MW between 21:00 and 04:59 (timestamps are IST: solar peaks at 12 h)

Shares (solar clipped at 0, the small negatives at night are sensor offset):
  vre_share  = (wind + solar) / demand
  sync_share = (thermal + hydro + gas + nuclear) / demand  where the split exists (to 2023-12),
               otherwise 1 - vre_share
Output: outputs/grid_india_hourly_clean.csv, figures/grid_india_clean.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "outputs", ROOT / "figures"
pd.set_option("display.width", 250)


def stuck(s, n=3):
    run = (s != s.shift()).cumsum()
    return s.groupby(run).transform("size") >= n


h = pd.read_csv(OUT / "grid_india_hourly.csv", parse_dates=["time"])
h["solar_MW"] = h.solar_MW.clip(lower=0)
has_split = h.thermal_MW.notna()
src = h[["thermal_MW", "hydro_MW", "gas_MW", "nuclear_MW", "wind_MW", "solar_MW"]].sum(axis=1)
ratio = np.where(has_split, src / h.demand_MW, h.total_gen_MW / h.demand_MW)
# neighbours: blank (zero) hours are ignored, so an outage cannot hide a partial hour next to it
med = h.demand_MW.where(h.demand_MW > 0).rolling(5, center=True, min_periods=2).median()
month_med = h.demand_MW.where(h.demand_MW > 0).groupby(h.time.dt.to_period("M")).transform("median")

h["flag_zero"] = h.demand_MW <= 0
h["flag_dropout"] = ((h.demand_MW - med).abs() / med > 0.15) & ~h.flag_zero
h["flag_low_demand"] = (h.demand_MW < 0.6 * month_med) & ~h.flag_zero
h["flag_balance"] = ((ratio < 0.95) | (ratio > 1.08)) & ~h.flag_zero
h["flag_stuck"] = (stuck(h.demand_MW) | stuck(h.wind_MW) | (stuck(h.total_gen_MW) & ~has_split)
                   | (stuck(h.solar_MW) & (h.solar_MW > 500)))
hod = h.time.dt.hour
h["flag_night_solar"] = ((hod >= 21) | (hod <= 4)) & (h.solar_MW > 1000)
flags = [c for c in h.columns if c.startswith("flag_")]
h["valid"] = ~h[flags].any(axis=1)

h["vre_share"] = (h.wind_MW + h.solar_MW) / h.demand_MW
h["sync_MW"] = h[["thermal_MW", "hydro_MW", "gas_MW", "nuclear_MW"]].sum(axis=1, min_count=4)
h["sync_share"] = np.where(has_split, h.sync_MW / h.demand_MW, 1 - h.vre_share)

print(f"hours {len(h)}; flagged per rule: {{" + ", ".join(f"{f[5:]}: {int(h[f].sum())}" for f in flags) + "}")
print(f"dropped (any rule): {int((~h.valid).sum())} ({(~h.valid).mean() * 100:.2f}%), kept {int(h.valid.sum())}")
c = h[h.valid].copy()
print("stuck hours by period: to 2023-12:", int(h[h.flag_stuck & has_split].shape[0]),
      " 2024-01 on:", int(h[h.flag_stuck & ~has_split].shape[0]))

ab = c[c.thermal_MW.notna()]
print(f"\ncheck, 2021-09..2023-12: sync_share vs 1 - vre_share: mean difference "
      f"{(ab.sync_share - (1 - ab.vre_share)).mean():+.4f}, max abs {(ab.sync_share - (1 - ab.vre_share)).abs().max():.4f}")
c["year"] = c.time.dt.year
print("\nclean hourly (wind + solar) / demand by year:")
print(c.groupby("year").vre_share.describe(percentiles=[.05, .5, .95, .99]).round(3).to_string())
print(f"max {c.vre_share.max():.3f} at {c.loc[c.vre_share.idxmax(), 'time']}, "
      f"min {c.vre_share.min():.3f} at {c.loc[c.vre_share.idxmin(), 'time']}")
print("\nclean synchronous share by year:")
print(c.groupby("year").sync_share.describe(percentiles=[.01, .5]).round(3).to_string())
top = c.nlargest(5, "vre_share")[["time", "demand_MW", "wind_MW", "solar_MW", "vre_share"]]
print("\ntop 5 clean renewable-share hours:")
print(top.round(3).to_string(index=False))

h.to_csv(OUT / "grid_india_hourly_clean.csv", index=False)

fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
w = c.set_index("time").resample("W").mean(numeric_only=True)
ax[0].plot(w.index, w.vre_share, label="weekly mean")
mx = c.set_index("time").vre_share.resample("W").max()
ax[0].plot(mx.index, mx, lw=0.8, label="weekly max hour")
ax[0].set(ylabel="(wind + solar) / demand", title="All-India variable renewable share (clean)")
ax[0].xaxis.set_major_locator(matplotlib.dates.YearLocator())
ax[0].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
ax[0].legend()
for y, g in c.groupby("year"):
    ax[1].plot(g.groupby(g.time.dt.hour).vre_share.mean(), label=str(y))
ax[1].set(xlabel="hour of day (IST)", ylabel="mean share", title="Average day by year")
ax[1].legend()
ax[2].hist(c.vre_share, bins=60, color="tab:green")
ax[2].set(xlabel="(wind + solar) / demand", ylabel="hours", title=f"All {len(c)} clean hours")
for a in ax:
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "grid_india_clean.png", dpi=150)
print("\nsaved outputs/grid_india_hourly_clean.csv and figures/grid_india_clean.png")
