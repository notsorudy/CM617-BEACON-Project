"""
BEACON: build SoH for the 86 Sandia cells from their periodic capacity checks.

How the Sandia test runs (seen in the data): a capacity check (full voltage window, 0.5C charge and
0.5C discharge) at the start, then the test condition, with a new check block roughly every 500
cycles. Normal cycles of partial-DoD cells move only part of the charge, so they are not capacity.

Check cycle = full voltage window (v_max and v_min within 0.05 V of the first check cycles) AND cycle
duration at least 90% of the first check cycles (0.5C/0.5C is the slowest cycle any condition uses).
Consecutive check cycles (index gap <= 10) form a block. Block capacity = median ah_d of the block.
SoH = block capacity / first block capacity.
Cycle 4 (switch-over from the first check to the test condition, logged as one long cycle with about
double the Ah) is excluded, as are rows with missing values. The last check of a cell with 3+ checks is
dropped if it falls more than 0.25 in SoH in one step (failed or stopped check; replicates do not show it).
A cell whose data starts after the first check (no reference capacity) is excluded.

Aging axes saved for every block: cycle_index, equivalent full cycles (cumulative discharged Ah /
first check capacity), and days since start.

Output: outputs/sandia_soh_points.csv (one row per check block), outputs/sandia_cells.csv (one row per
cell), figures/sandia_soh.png
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIR, OUT, FIG = ROOT / "data" / "sandia" / "cycle", ROOT / "outputs", ROOT / "figures"
pd.set_option("display.width", 250)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})


def parse(cid):
    m = re.match(r"SNL_18650_(\w+)_(\d+)C_(\d+)-(\d+)_([\d.]+)/([\d.]+)C_(\w)", cid)
    chem, t, lo, hi, cc, cd, rep = m.groups()
    return dict(chem=chem, temp_C=float(t), soc_min=float(lo), soc_max=float(hi), dod=float(hi) - float(lo),
                c_charge=float(cc), c_discharge=float(cd), replicate=rep)


def cell_points(df):
    d = df.dropna(subset=["test_time", "ah_d", "v_max", "v_min"]).copy()
    d = d[d.cycle_index != 4].sort_values("cycle_index").reset_index(drop=True)
    d["dur"] = d.test_time.shift(-1) - d.test_time
    ref = d[d.cycle_index <= 3]
    if ref.empty:   # data starts later (first check missing in the archive): no reference capacity
        return None, dict(n_rows=len(df), first_cycle=int(df.cycle_index.min()))
    vmax0, vmin0, dur0 = ref.v_max.median(), ref.v_min.median(), ref.dur.median()
    q0 = ref.ah_d.median()
    d["efc"] = d.ah_d.fillna(0).cumsum() / q0
    d["days"] = (d.test_time - d.test_time.iloc[0]) / 86400
    chk = d[(d.v_max >= vmax0 - 0.05) & (d.v_min <= vmin0 + 0.05) & (d.dur >= 0.9 * dur0)].copy()
    chk["block"] = (chk.cycle_index.diff() > 10).cumsum()
    pts = chk.groupby("block").agg(cycle_index=("cycle_index", "first"), n_check_cycles=("ah_d", "size"),
                                   cap_Ah=("ah_d", "median"), efc=("efc", "first"), days=("days", "first"),
                                   v_c_mean=("v_c_mean", "median"), v_d_mean=("v_d_mean", "median"))
    pts["SoH"] = pts.cap_Ah / pts.cap_Ah.iloc[0]
    # end-of-test artifact: the LAST check falls more than 0.25 in one step (failed or stopped check)
    # (only with 3+ checks, so there is an earlier trend; a 2-check cell's single drop can be real)
    dropped_last = len(pts) >= 3 and (pts.SoH.iloc[-2] - pts.SoH.iloc[-1]) > 0.25
    if dropped_last:
        pts = pts.iloc[:-1]
    return pts.reset_index(drop=True), dict(dropped_last_point=bool(dropped_last),vmax0=vmax0, vmin0=vmin0, check_dur_h=dur0 / 3600, q0=q0,
                                            n_rows=len(df), n_used=len(d), n_check=len(chk))


def main():
    files = sorted(DIR.glob("SNL_*.csv"))
    pts_all, cells, excluded = [], [], []
    for f in files:
        df = pd.read_csv(f)
        cid = df.cell_id.iloc[0]
        pts, info = cell_points(df)
        meta = parse(cid)
        if pts is None:
            print(f"EXCLUDED {cid}: data starts at cycle {info['first_cycle']}, no first capacity check")
            excluded.append(dict(cell_id=cid, reason=f"data starts at cycle {info['first_cycle']}, no first check"))
            continue
        pts.insert(0, "cell_id", cid)
        for k, v in meta.items():
            pts[k] = v
        pts_all.append(pts)
        cells.append(dict(cell_id=cid, **meta, **info, n_blocks=len(pts), last_cycle=int(df.cycle_index.max()),
                          final_SoH=pts.SoH.iloc[-1], final_efc=pts.efc.iloc[-1], final_days=pts.days.iloc[-1]))
    P = pd.concat(pts_all, ignore_index=True)
    C = pd.DataFrame(cells)
    P.to_csv(OUT / "sandia_soh_points.csv", index=False)
    C.to_csv(OUT / "sandia_cells.csv", index=False)
    pd.DataFrame(excluded).to_csv(OUT / "sandia_excluded.csv", index=False)

    print(f"cells {len(C)}: {C.chem.value_counts().to_dict()}; check-block points {len(P)}")
    print("rows dropped (NaN or cycle 4) per cell: max", int((C.n_rows - C.n_used).max()))
    print("last point dropped as end-of-test artifact:", C[C.dropped_last_point].cell_id.tolist())
    print("\nfirst-check values by chemistry (median): ")
    print(C.groupby("chem")[["q0", "vmax0", "vmin0", "check_dur_h"]].median().round(3).to_string())
    print("\nblocks per cell: min %d, median %d, max %d" % (C.n_blocks.min(), C.n_blocks.median(), C.n_blocks.max()))
    print("cells with fewer than 3 blocks:", C[C.n_blocks < 3].cell_id.tolist())
    print("\nfinal SoH by chemistry and condition:")
    print(C.groupby(["chem", "temp_C", "dod", "c_discharge"]).agg(
        cells=("cell_id", "size"), blocks=("n_blocks", "median"), final_SoH=("final_SoH", "median"),
        min_SoH=("final_SoH", "min"), efc=("final_efc", "median"), days=("final_days", "median")).round(3).to_string())
    print("\nSoH range overall: %.3f to %.3f" % (P.SoH.min(), P.SoH.max()))
    print("points with SoH > 1.02 (recovery or noise):", int((P.SoH > 1.02).sum()))
    jump = P.groupby("cell_id").SoH.diff()
    print("block-to-block SoH rises > 0.02:", int((jump > 0.02).sum()))

    fig, ax = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    for a, chem in zip(ax, ["LFP", "NMC", "NCA"]):
        for cid, g in P[P.chem == chem].groupby("cell_id"):
            m = g.iloc[0]
            col = {15.0: "tab:blue", 25.0: "tab:green", 35.0: "tab:red"}[m.temp_C]
            ls = {100.0: "-", 60.0: "--", 20.0: ":"}[m.dod]
            a.plot(g.efc, g.SoH, color=col, ls=ls, marker="o", ms=2.5, lw=1)
        a.set(title=f"{chem} ({(C.chem == chem).sum()} cells)", xlabel="equivalent full cycles")
    ax[0].set_ylabel("SoH (capacity check / first check)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=c, label=l) for c, l in [("tab:blue", "15 C"), ("tab:green", "25 C"), ("tab:red", "35 C")]]
    h += [Line2D([], [], color="k", ls=s, label=l) for s, l in [("-", "DoD 100%"), ("--", "DoD 60%"), (":", "DoD 20%")]]
    ax[0].legend(handles=h, fontsize=8)
    fig.suptitle("Sandia cells (Preger et al. 2020, batteryarchive.org): SoH from periodic capacity checks")
    fig.tight_layout()
    fig.savefig(FIG / "sandia_soh.png", dpi=150)
    print("saved outputs/sandia_soh_points.csv, outputs/sandia_cells.csv, figures/sandia_soh.png")


if __name__ == "__main__":
    main()
