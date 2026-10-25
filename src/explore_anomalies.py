"""BEACON: locate anomalous summary values (prints only)."""
import numpy as np
import pandas as pd

from load_clean import load_clean

cells, split, _ = load_clean()
rows = []
for k, c in cells.items():
    s = c["summary"]
    df = pd.DataFrame({f: s[f] for f in s})
    df["cell"] = k
    df["idx"] = np.arange(len(df))
    # rolling median of the neighbours, centred
    for f in ["QD", "IR"]:
        med = df[f].replace(0, np.nan).rolling(11, center=True, min_periods=3).median()
        df[f + "_dev"] = df[f] - med
    rows.append(df)
d = pd.concat(rows, ignore_index=True)
pd.set_option("display.width", 200)
cols = ["cell", "idx", "cycle", "QD", "QC", "IR", "Tavg", "Tmin", "Tmax", "chargetime", "QD_dev", "IR_dev"]
print("Tavg==0 or Tmin==0 rows:", int(((d.Tavg == 0) | (d.Tmin == 0)).sum()))
print(d[(d.Tavg == 0) | (d.Tmin == 0)][cols].groupby("cell").size().to_dict())
print("\nchargetime == 0:", int((d.chargetime == 0).sum()), " chargetime > 30 min:", int((d.chargetime > 30).sum()))
print(d[d.chargetime > 30][cols].to_string())
print("\nQD spikes |dev| > 0.02 Ah (excluding QD==0):")
print(d[(d.QD_dev.abs() > 0.02) & (d.QD > 0)][cols].to_string())
print("\nIR spikes |dev| > 0.001 ohm (excluding IR==0):")
sp = d[(d.IR_dev.abs() > 0.001) & (d.IR > 0)]
print(len(sp)); print(sp[cols].head(40).to_string())
print("\nchargetime percentiles by batch:")
d["batch"] = d.cell.str[:2]
print(d[d.chargetime > 0].groupby("batch").chargetime.describe(percentiles=[.01, .5, .99]))
print("\nIR at merge point for b1c0..b1c4 (last 3 of batch1 part, first 3 of batch2 part):")
for k, b2len in zip(["b1c0", "b1c1", "b1c2", "b1c3", "b1c4"], [662, 981, 1060, 208, 482]):
    x = d[d.cell == k]
    n1 = len(x) - None if False else None
    print(k, x[["cycle", "QD", "IR", "Tavg"]].iloc[[0]].values.tolist())
