"""BEACON: look at suspicious hours in the Grid-India hourly table (prints only)."""
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs"
pd.set_option("display.width", 250)
h = pd.read_csv(OUT / "grid_india_hourly.csv", parse_dates=["time"])
cols = ["time", "demand_MW", "thermal_MW", "hydro_MW", "gas_MW", "nuclear_MW", "wind_MW", "solar_MW", "total_gen_MW",
        "n_samples"]
print("all-zero demand hours:")
print(h[h.demand_MW <= 0][cols].to_string(index=False))
print("\nnuclear > 8000 MW:")
print(h[h.nuclear_MW > 8000][cols].to_string(index=False))
print("\nwind > 27000 MW (top 10):")
print(h[h.wind_MW > 27000].nlargest(10, "wind_MW")[cols].to_string(index=False))
h["share"] = (h.wind_MW.clip(lower=0) + h.solar_MW.clip(lower=0)) / h.demand_MW
print("\ntop 12 renewable share hours:")
print(h.nlargest(12, "share")[cols + ["share"]].round(3).to_string(index=False))
d = h.set_index("time").demand_MW
jump = (d - d.rolling(5, center=True).median()).abs() / d
print("\nhours where demand differs > 15% from its 5-hour rolling median:")
print(h.set_index("time")[jump > 0.15].reset_index()[cols].to_string(index=False))
src = h.thermal_MW + h.hydro_MW + h.gas_MW + h.nuclear_MW + h.wind_MW + h.solar_MW.clip(lower=0)
r = src / h.demand_MW
print("\nperiods A+B hours where sources / demand outside 0.95..1.08:", int(((r < 0.95) | (r > 1.08)).sum()))
print(h[(r < 0.95) | (r > 1.08)][cols].head(20).to_string(index=False))
c = h[h.total_gen_MW.notna()]
rc = c.total_gen_MW / c.demand_MW
print("\nperiod C hours where total_gen / demand outside 0.95..1.08:", int(((rc < 0.95) | (rc > 1.08)).sum()))
print(c[(rc < 0.95) | (rc > 1.08)][cols].head(20).to_string(index=False))
