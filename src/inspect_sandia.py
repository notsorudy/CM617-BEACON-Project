"""BEACON: first look at the downloaded Sandia per-cycle CSVs (prints only)."""
from pathlib import Path

import numpy as np
import pandas as pd

DIR = Path(__file__).resolve().parents[1] / "data" / "sandia" / "cycle"
pd.set_option("display.width", 250)
files = sorted(DIR.glob("SNL_*.csv"))
print(f"{len(files)} cell files")
d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
d["chem"] = d.cell_id.str.split("_").str[2]
d["temp"] = d.cell_id.str.split("_").str[3]
d["dod"] = d.cell_id.str.split("_").str[4]
d["rate"] = d.cell_id.str.split("_").str[5].str.split("/").str[1]
print("columns:", list(d.columns))
print("NaN per column:", d.isna().sum().to_dict())

per = d.groupby("cell_id").agg(chem=("chem", "first"), dod=("dod", "first"), rate=("rate", "first"),
                               cycles=("cycle_index", "size"), last=("cycle_index", "max"),
                               ahd_med=("ah_d", "median"), ahd_max=("ah_d", "max"),
                               time_days=("test_time", lambda x: x.max() / 86400))
print("\ncells per chemistry:", per.chem.value_counts().to_dict())
print("cycles per cell: min %d, median %d, max %d" % (per.cycles.min(), per.cycles.median(), per.cycles.max()))
print(per.groupby(["chem", "dod"]).agg(cells=("cycles", "size"), cycles_med=("cycles", "median"),
                                       ahd_med=("ahd_med", "median"), days_med=("time_days", "median")).round(2))

# look at one cell per chemistry / DoD to understand the cycle pattern
for cid in ["SNL_18650_LFP_25C_0-100_0.5/1C_a", "SNL_18650_LFP_25C_20-80_0.5/0.5C_a",
            "SNL_18650_NMC_25C_0-100_0.5/2C_a", "SNL_18650_NCA_25C_40-60_0.5/0.5C_a"]:
    x = d[d.cell_id == cid]
    if x.empty:
        print(f"\n=== {cid}: not downloaded yet")
        continue
    print(f"\n=== {cid}: {len(x)} cycles")
    print(x[["cycle_index", "test_time", "ah_c", "ah_d", "e_d", "v_c_mean", "v_d_mean", "v_max", "v_min"]]
          .head(14).round(4).to_string(index=False))
    q = x.ah_d
    hi = q > 0.8 * q.quantile(0.98)
    print(f"cycles with ah_d > 0.8 x (98th pct): {int(hi.sum())}; their cycle_index (first 20):",
          x.cycle_index[hi].astype(int).tolist()[:20])
    gaps = np.diff(x.cycle_index[hi].to_numpy())
    print("gaps between those cycles (first 20):", gaps[:20].astype(int).tolist())
