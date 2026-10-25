"""BEACON: review the Kutch Renewables.ninja data for the solar-only / wind-only study (prints only).
What matters for an inverter-only system with a grid-forming battery:
  when the plant produces, how much, how much it drops from one hour to the next (stress for the battery),
  how often it is near zero, and the site temperature (an input to battery aging)."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
IST = pd.Timedelta(hours=5, minutes=30)
pv = pd.read_csv(DATA / "ninja_pv_23.2500_69.6703_uncorrected.csv", comment="#", parse_dates=["time"])
wd = pd.read_csv(DATA / "ninja_wind_23.2500_69.6703_uncorrected.csv", comment="#", parse_dates=["time"])
pv["t"] = pv.time + IST
wd["t"] = wd.time + IST
pd.set_option("display.width", 200)

for name, df in [("SOLAR", pv), ("WIND", wd)]:
    cf = df.electricity
    drop = (cf.shift(1) - cf).clip(lower=0)           # hour-to-hour fall in output (fraction of capacity)
    print(f"\n===== {name} (capacity factor, 1 = full plant output) =====")
    print(f"annual mean {cf.mean():.3f}; hours with output > 5%: {int((cf > 0.05).sum())} of {len(cf)} "
          f"({100 * (cf > 0.05).mean():.1f}%); > 50%: {int((cf > 0.5).sum())}")
    print("output percentiles (hours with output > 5%):",
          cf[cf > 0.05].quantile([.1, .25, .5, .75, .9, .99]).round(3).to_dict())
    print(f"hour-to-hour drop: 95th pct {drop.quantile(.95):.3f}, 99th {drop.quantile(.99):.3f}, max {drop.max():.3f}")
    longest = (cf <= 0.05).astype(int)
    runs = longest.groupby((longest != longest.shift()).cumsum()).sum()
    print(f"longest stretch with output <= 5%: {int(runs.max())} hours")
    m = df.groupby(df.t.dt.month).electricity.mean().round(3)
    print("monthly mean:", m.to_dict())
    h = df.groupby(df.t.dt.hour).electricity.mean().round(3)
    print("mean by IST hour:", h.to_dict())

print("\n===== SITE AIR TEMPERATURE (from the solar file, deg C) =====")
T = pv.temperature
print(f"min {T.min():.1f}, mean {T.mean():.1f}, max {T.max():.1f}; hours >= 35 C: {int((T >= 35).sum())}; "
      f"hours >= 40 C: {int((T >= 40).sum())}")
print("monthly mean:", pv.groupby(pv.t.dt.month).temperature.mean().round(1).to_dict())
print("mean temperature when solar > 50%:", round(pv.temperature[pv.electricity > 0.5].mean(), 1),
      " when wind > 50%:", round(pv.temperature[wd.electricity > 0.5].mean(), 1))

print("\n===== SIMPLE BATTERY DUTY HINT (battery smooths the plant to its daily mean, 1 kW plant) =====")
for name, df in [("solar", pv), ("wind", wd)]:
    day = df.groupby(df.t.dt.date).electricity
    swing = (df.electricity - day.transform("mean"))      # surplus (+) / deficit (-) vs daily mean
    soc_range = df.assign(s=swing).groupby(df.t.dt.date).s.apply(lambda x: x.cumsum().max() - x.cumsum().min())
    print(f"{name}: daily energy shifted to flatten output, median {soc_range.median():.2f} kWh per kW of plant, "
          f"90th pct {soc_range.quantile(.9):.2f}, max {soc_range.max():.2f}")
