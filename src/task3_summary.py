"""BEACON Task 3: compact tables from outputs/task3_cases.csv (prints only)."""
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs"
c = pd.read_csv(OUT / "task3_cases.csv")
pd.set_option("display.width", 250)

for tr in [10.0, 1.0]:
    for ev in [0.10, 0.05]:
        s = c[(c.t_res == tr) & (c.event == ev)]
        print(f"\n### energy reserve {tr} s, {int(ev * 100)}% loss: f_nadir (Hz) [S = cap saturates, E = energy runs out]")
        s = s.assign(cell=s.f_nadir_Hz.map("{:.3f}".format) + s.saturated.map({True: " S", False: ""})
                     + s.energy_limit_hit.map({True: " E", False: ""}))
        print(s.pivot_table(index=["curve", "soh"], columns="demand_actual", values="cell", aggfunc="first")
              .to_string())
        print("RoCoF 500 ms (Hz/s):")
        print(s.pivot_table(index=["curve", "soh"], columns="demand_actual", values="rocof_500ms_Hz_s").round(3)
              .to_string())

print("\nSettling time (s), 10% loss, reserve 10 s:")
s = c[(c.t_res == 10) & (c.event == 0.10)]
print(s.pivot_table(index=["curve", "soh"], columns="demand_actual", values="settling_s").round(2).to_string())

print("\nCases where aging makes no difference vs the fresh battery (nadir and RoCoF within 0.001):")
a = c[c.soh < 1.0]
print(a.groupby(["t_res", "event", "demand_actual"]).aging_no_difference.agg(["sum", "size"]).to_string())

print("\nLargest nadir change vs fresh (Hz), per curve, SoH <= 0.82, data-backed only (SoH >= 0.76):")
b = c[(c.soh == 0.82)]
print(b.pivot_table(index=["t_res", "event", "curve"], columns="demand_actual", values="d_nadir_vs_fresh_Hz")
      .round(4).to_string())
print("\nEnergy: peak used / available, reserve 1 s, 10% loss:")
e = c[(c.t_res == 1.0) & (c.event == 0.10) & (c.curve == "quadratic")]
print(e.assign(ratio=e.energy_used_peak_pu_s / e.energy_avail_pu_s)
      .pivot_table(index="soh", columns="demand_actual", values="ratio").round(3).to_string())
e = c[(c.t_res == 10.0) & (c.event == 0.10) & (c.curve == "quadratic")]
print("reserve 10 s, max used/available:", round((e.energy_used_peak_pu_s / e.energy_avail_pu_s).max(), 3))
