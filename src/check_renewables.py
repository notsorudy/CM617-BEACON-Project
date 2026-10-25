"""BEACON: checks on the Renewables.ninja solar and wind files (prints + one figure).
Completeness, time zone, value ranges, capacity factors, daily and seasonal shape."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT / "data", ROOT / "figures"
FILES = {"pv": "ninja_pv_23.2500_69.6703_uncorrected.csv", "wind": "ninja_wind_23.2500_69.6703_uncorrected.csv"}
IST = pd.Timedelta(hours=5, minutes=30)

d = {}
for k, f in FILES.items():
    lines = open(DATA / f, encoding="utf-8").read().splitlines()
    meta = json.loads(lines[2].lstrip("# "))
    df = pd.read_csv(DATA / f, comment="#")
    df["time"] = pd.to_datetime(df.time)
    d[k] = df
    p = meta["params"]
    print(f"=== {k}: {f}")
    print("  params:", {x: p[x] for x in p if x not in ("header", "raw")})
    print("  units:", meta["units"])
    t = df.time
    gaps = t.diff().dropna()
    print(f"  rows {len(df)}, from {t.min()} to {t.max()} (UTC), steps not 1 h: {int((gaps != pd.Timedelta('1h')).sum())}, "
          f"duplicate times: {int(t.duplicated().sum())}")
    print(f"  missing values per column: {df.isna().sum().to_dict()}")
    for c in df.columns[1:]:
        v = df[c]
        print(f"  {c:20s} min {v.min():8.3f}  mean {v.mean():8.3f}  max {v.max():8.3f}  negatives {int((v < 0).sum())}")
    print(f"  annual capacity factor (mean electricity / 1 kW): {df.electricity.mean():.3f}")

pv, wd = d["pv"], d["wind"]
print("\nsame timestamps in both files:", bool((pv.time.values == wd.time.values).all()))

# time zone: solar peak hour in UTC and in IST
pv["hour_utc"] = pv.time.dt.hour
pv["hour_ist"] = (pv.time + IST).dt.hour + (pv.time + IST).dt.minute / 60
prof_utc = pv.groupby("hour_utc").electricity.mean()
print(f"\nsolar: mean output peaks at {prof_utc.idxmax():02d}:00 UTC = {prof_utc.idxmax() + 5.5:.1f} h IST "
      "(should be around 12 to 13 h local)")
night = pv[(pv.hour_utc >= 15) & (pv.hour_utc <= 22)]   # 20:30 to 03:30 IST
print(f"solar between 20:30 and 04:30 IST: max {night.electricity.max():.3f} kW (should be 0)")

# seasonal shape
for k, df in [("pv", pv), ("wind", wd)]:
    m = df.groupby((df.time + IST).dt.month).electricity.mean().round(3)
    print(f"{k} monthly capacity factor (IST months): {m.to_dict()}")
wd_hour = wd.groupby((wd.time + IST).dt.hour).electricity.mean()
print(f"wind mean by IST hour: min {wd_hour.min():.3f} at {wd_hour.idxmin()} h, max {wd_hour.max():.3f} at {wd_hour.idxmax()} h")

# wind speed vs electricity consistent with a power curve
b = wd.groupby(pd.cut(wd.wind_speed, [0, 3, 4, 6, 8, 10, 12, 14, 25, 40]), observed=True).electricity.agg(["mean", "max", "size"])
print("\nwind power vs wind speed (power-curve check):")
print(b.round(3).to_string())
print("hours both solar and wind below 5% of capacity:", int(((pv.electricity < 0.05) & (wd.electricity < 0.05)).sum()))

# figure
fig, ax = plt.subplots(2, 2, figsize=(14, 8))
loc = pv.time + IST
ax[0, 0].plot(loc, pv.electricity.rolling(24 * 7, center=True).mean(), label="solar (7-day mean)")
ax[0, 0].plot(loc, wd.electricity.rolling(24 * 7, center=True).mean(), label="wind (7-day mean)")
ax[0, 0].set(ylabel="capacity factor", title="Kutch, Gujarat (23.25 N, 69.67 E), 2025, Renewables.ninja MERRA-2")
ax[0, 0].legend()
for k, df, c in [("solar", pv, "tab:blue"), ("wind", wd, "tab:orange")]:
    h = df.groupby((df.time + IST).dt.hour).electricity.mean()
    ax[0, 1].plot(h.index, h.values, "o-", color=c, label=k)
ax[0, 1].set(xlabel="hour of day (IST, rounded)", ylabel="mean capacity factor", title="Average day")
ax[0, 1].legend()
wk = (loc >= "2025-06-01") & (loc < "2025-06-08")
ax[1, 0].plot(loc[wk], pv.electricity[wk], label="solar")
ax[1, 0].plot(loc[wk], wd.electricity[wk], label="wind")
ax[1, 0].set(ylabel="capacity factor", title="One week, 1 to 7 June 2025 (IST)")
ax[1, 0].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
ax[1, 0].legend()
ax[1, 1].scatter(wd.wind_speed, wd.electricity, s=2, alpha=0.3)
ax[1, 1].set(xlabel="wind speed at 100 m (m/s)", ylabel="wind capacity factor", title="Wind power curve check")
for a in ax.flat:
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "renewables_data_check.png", dpi=150)
print("saved figures/renewables_data_check.png")
