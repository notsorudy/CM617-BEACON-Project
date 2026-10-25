"""BEACON: how do the per-cycle summary values relate to the within-cycle time points?
Compares summary QD, QC, Tmax, Tmin, Tavg, chargetime with values computed from the raw
time series, for every cycle of a few batch2 cells (prints only)."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
full = pickle.load(open(DATA / "batch2.pkl", "rb"))

rows = []
for k in ["b2c2", "b2c10", "b2c30", "b2c47"]:
    s, cyc = full[k]["summary"], full[k]["cycles"]
    keys = sorted(cyc, key=int)
    print(f"{k}: {len(s['cycle'])} summary rows, {len(keys)} within-cycle records, "
          f"summary cycle numbers {s['cycle'][0]:.0f}..{s['cycle'][-1]:.0f}, cycle keys {keys[0]}..{keys[-1]}")
    for i in range(1, len(s["cycle"])):
        c = cyc.get(str(i))
        if c is None or len(c["t"]) < 10:
            continue
        I, t, T = c["I"], c["t"], c["T"]
        charging = I > 0.01
        rows.append(dict(cell=k, idx=i,
                         QD_sum=s["QD"][i], QD_raw=c["Qd"].max(),
                         QC_sum=s["QC"][i], QC_raw=c["Qc"].max(),
                         Tmax_sum=s["Tmax"][i], Tmax_raw=T.max(),
                         Tmin_sum=s["Tmin"][i], Tmin_raw=T.min(),
                         Tavg_sum=s["Tavg"][i], Tavg_raw=T.mean(),
                         ct_sum=s["chargetime"][i],
                         ct_raw=(t[charging].max() - t[charging].min()) if charging.any() else np.nan,
                         n_points=len(t), duration_min=t.max() - t.min()))
d = pd.DataFrame(rows)
print(f"\ncompared {len(d)} cycles")
for f in ["QD", "QC", "Tmax", "Tmin", "Tavg", "ct"]:
    diff = d[f + "_sum"] - d[f + "_raw"]
    print(f"{f:5s} summary - raw: median {diff.median():+.5f}, 95% of |diff| below {diff.abs().quantile(.95):.5f}")
print(f"\npoints per cycle: median {d.n_points.median():.0f}, min {d.n_points.min()}, max {d.n_points.max()}")
print(f"cycle duration (min): median {d.duration_min.median():.1f}")
print("\nexample, b2c2 cycle index 100:")
print(d[(d.cell == "b2c2") & (d.idx == 100)].T.to_string())
