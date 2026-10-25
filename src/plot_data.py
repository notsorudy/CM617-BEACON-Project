"""BEACON: data overview plots (capacity fade and IR vs cycle)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

df = pd.read_csv(ROOT / "outputs" / "cycle_table.csv")
raw = pd.read_csv(ROOT / "outputs" / "cycle_table_with_flags.csv")

# a few cells: short, medium and long life, from each batch
pick = ["b1c0", "b1c20", "b1c45", "b2c1", "b2c47", "b3c6", "b3c38"]
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for k in pick:
    d = df[df.cell_id == k]
    lab = f"{k} ({d.policy.iloc[0].replace('-newstructure', '')})"
    ax[0].plot(d.cycle, d.QD_Ah, lw=1.2, label=lab)
    ax[1].plot(d.cycle, d.IR_ohm * 1000, lw=1.0, label=lab)
ax[0].axhline(0.88, color="k", ls="--", lw=1, label="80% of 1.1 Ah (EOL)")
ax[0].set(xlabel="Cycle number", ylabel="Discharge capacity (Ah)", title="Capacity fade, selected cells")
ax[1].set(xlabel="Cycle number", ylabel="Internal resistance (mOhm)", title="Internal resistance, selected cells")
ax[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "data_capacity_IR_selected_cells.png", dpi=150)

# all cells, coloured by batch
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
for k, d in df.groupby("cell_id"):
    c = colors[d.batch.iloc[0]]
    ax[0, 0].plot(d.cycle, d.QD_Ah, color=c, lw=0.5, alpha=0.6)
    ax[0, 1].plot(d.cycle, d.IR_ohm * 1000, color=c, lw=0.4, alpha=0.5)
    ax[1, 0].plot(d.cycle, d.SoH_Q, color=c, lw=0.5, alpha=0.6)
    ax[1, 1].plot(d.cycle, d.R_ratio, color=c, lw=0.4, alpha=0.5)
for b, c in colors.items():
    ax[0, 0].plot([], [], color=c, label=f"batch {b}")
ax[0, 0].legend()
ax[0, 0].set(xlabel="Cycle number", ylabel="Discharge capacity (Ah)", title="All 124 cleaned cells")
ax[0, 1].set(xlabel="Cycle number", ylabel="Internal resistance (mOhm)")
ax[1, 0].set(xlabel="Cycle number", ylabel="SoH_Q = QD / Q0")
ax[1, 1].set(xlabel="Cycle number", ylabel="R / R0")
fig.tight_layout()
fig.savefig(FIG / "data_all_cells_overview.png", dpi=150)

# before vs after cleaning for one cell with many artifacts
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for a, col, unit, scale in [(ax[0], "QD_Ah", "Discharge capacity (Ah)", 1), (ax[1], "IR_ohm", "IR (mOhm)", 1000)]:
    for k in ["b1c2", "b2c44"]:
        r = raw[raw.cell_id == k]
        bad = r[~r.valid]
        good = r[r.valid]
        a.plot(good.cycle, good[col] * scale, lw=0.8, label=f"{k} kept")
        a.scatter(bad.cycle, bad[col] * scale, color="red", s=18, zorder=3)
    a.scatter([], [], color="red", s=18, label="dropped rows")
    a.set(xlabel="Cycle number", ylabel=unit)
    a.legend(fontsize=8)
ax[0].set_ylim(0.85, 1.6)
ax[0].set_title("Cleaning: rows dropped (red)")
fig.tight_layout()
fig.savefig(FIG / "data_cleaning_example.png", dpi=150)
print("saved 3 figures")
