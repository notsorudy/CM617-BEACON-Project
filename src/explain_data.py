"""BEACON: facts for explaining the data (prints only).
1. What the removed cells look like.
2. What one summary row and one within-cycle record look like, and how big each is."""
import pickle
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"
REMOVED = {1: ["b1c8", "b1c10", "b1c12", "b1c13", "b1c22"], 2: ["b2c7", "b2c8", "b2c9", "b2c15", "b2c16"],
           3: ["b3c37", "b3c2", "b3c23", "b3c32", "b3c42", "b3c43"]}

print("=== cells removed or merged by Load Data.ipynb ===")
for b, keys in REMOVED.items():
    d = pickle.load(open(DATA / f"batch{b}_summary.pkl", "rb"))
    for k in keys:
        s = d[k]["summary"]
        qd = s["QD"][s["QD"] > 0]
        q0 = np.median(qd[1:11])
        cl = float(np.ravel(d[k]["cycle_life"])[0]) if np.size(d[k]["cycle_life"]) else np.nan
        print(f"{k:6s} cycles {len(s['cycle']):5d}  listed cycle_life {cl:7.0f}  QD start {q0:.3f}  QD last {qd[-1]:.3f}"
              f"  min SoH {qd.min() / q0:.3f}  IR zeros {int(np.sum(s['IR'] == 0))}  policy {d[k]['charge_policy']}")

print("\n=== one summary row: cell b2c2, cycle 100 ===")
d = pickle.load(open(DATA / "batch2_summary.pkl", "rb"))
s = d["b2c2"]["summary"]
i = int(np.where(s["cycle"] == 100)[0][0])
for f in s:
    print(f"  {f:10s} {s[f][i]:.5f}")
print(f"  cycle_life {float(np.ravel(d['b2c2']['cycle_life'])[0]):.0f}, policy {d['b2c2']['charge_policy']}")
tot = sum(len(c["summary"]["cycle"]) for c in d.values())
print(f"batch2 summary: {len(d)} cells, {tot} rows in total")
del d

print("\n=== within-cycle data, batch2 (loading the 2 GB file) ===")
full = pickle.load(open(DATA / "batch2.pkl", "rb"))
cyc = full["b2c2"]["cycles"]["100"]
for f, v in cyc.items():
    print(f"  {f:6s} {len(v):5d} points, first {v[0]:.4f}, last {v[-1]:.4f}")
n_cycles = sum(len(c["cycles"]) for c in full.values())
n_raw = sum(len(cy["t"]) for c in full.values() for cy in c["cycles"].values())
n_all = sum(sum(len(v) for v in cy.values()) for c in full.values() for cy in c["cycles"].values())
print(f"batch2: {n_cycles} cycles, {n_raw} raw time points (t), {n_all} numbers in total across all within-cycle fields")
