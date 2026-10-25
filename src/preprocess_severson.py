"""
BEACON: preprocessing for the Severson/TRI battery dataset.
Builds the per-cycle table for all cleaned cells (after the Load Data.ipynb rules).

Started from the HANDOFF section 11 script. Fixes and changes:
- Works on the cleaned, merged summary data (load_clean.py), not one raw pkl.
- Batch1 cycle 1 is all zeros (QD, IR, T, chargetime). The old script used cycle index 1
  as reference, which happened to avoid it, but it kept the zero row in the table.
- References are robust: Q0 and R0 = median of the first 10 valid rows from cycle 2 on
  (normally cycles 2 to 11), instead of one single cycle.
- Bad rows are flagged and dropped (rules below) instead of being kept.
- Adds charging policy features parsed from the policy string, and the split label.
- Windows-safe paths. Output goes to outputs/cycle_table.csv.

Row cleaning rules (row is dropped if any is true):
  zero_row     : QD == 0 or Tavg == 0 (batch1 cycle 1 logging artifact)
  ir_zero      : IR == 0 (IR missing, 62 rows in total)
  pause        : chargetime > 60 min (tester pause or restart, not a normal cycle)
  qd_spike     : |QD - rolling median of 11 cycles| > 0.05 Ah
  ir_spike     : |IR - rolling median of 11 cycles| > 0.0015 ohm
  ir_stuck     : exactly the same IR value for 5 or more cycles in a row (held value,
                 not a measurement; b1c18 cycles 2 to ~53)
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from load_clean import load_clean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

NOMINAL_AH = 1.1
QD_SPIKE_AH = 0.05
IR_SPIKE_OHM = 0.0015
PAUSE_MIN = 60.0
STUCK_RUN = 5
REF_START, REF_N = 2, 10


def parse_policy(p):
    """'5.4C(60%)-3C' -> C1=5.4, Q1=60, C2=3. Also handles '4C(31%)-5' and '-newstructure'."""
    p = p.replace("-newstructure", "")
    m = re.match(r"([\d.]+)C\(([\d.]+)%\)-([\d.]+)C?", p)
    if not m:
        raise ValueError(f"cannot parse policy {p!r}")
    c1, q1, c2 = float(m.group(1)), float(m.group(2)), float(m.group(3))
    # time to charge 0 -> 80% SOC in minutes (two-step constant current)
    t80 = 60 * (q1 / 100 / c1 + max(0.0, 80 - q1) / 100 / c2)
    return dict(C1=c1, Q1=q1, C2=c2, t_0_80_min=t80, Cavg_0_80=0.8 / (t80 / 60))


def cell_table(cell_id, cell):
    s = cell["summary"]
    df = pd.DataFrame({
        "cycle": s["cycle"], "QD_Ah": s["QD"], "QC_Ah": s["QC"], "IR_ohm": s["IR"],
        "T_avg": s["Tavg"], "T_max": s["Tmax"], "T_min": s["Tmin"], "chargetime_min": s["chargetime"],
    })
    df.insert(0, "cell_id", cell_id)
    df.insert(1, "batch", int(cell_id[1]))
    df["cycle_life"] = float(np.ravel(cell["cycle_life"])[0])
    df["policy"] = cell["charge_policy"]
    for k, v in parse_policy(cell["charge_policy"]).items():
        df[k] = v

    df["flag_zero_row"] = (df.QD_Ah == 0) | (df.T_avg == 0)
    df["flag_ir_zero"] = df.IR_ohm == 0
    df["flag_pause"] = df.chargetime_min > PAUSE_MIN
    qd = df.QD_Ah.where(~df.flag_zero_row)
    ir = df.IR_ohm.where(~df.flag_ir_zero & ~df.flag_zero_row)
    qd_med = qd.rolling(11, center=True, min_periods=3).median()
    ir_med = ir.rolling(11, center=True, min_periods=3).median()
    df["flag_qd_spike"] = (qd - qd_med).abs() > QD_SPIKE_AH
    df["flag_ir_spike"] = (ir - ir_med).abs() > IR_SPIKE_OHM
    run_id = (df.IR_ohm != df.IR_ohm.shift()).cumsum()
    run_len = df.groupby(run_id).IR_ohm.transform("size")
    df["flag_ir_stuck"] = (run_len >= STUCK_RUN) & (df.IR_ohm > 0)
    flags = [c for c in df.columns if c.startswith("flag_")]
    df["valid"] = ~df[flags].any(axis=1)

    ref = df[df.valid & (df.cycle >= REF_START)].head(REF_N)
    q0, r0 = ref.QD_Ah.median(), ref.IR_ohm.median()
    df["Q0_Ah"], df["R0_ohm"] = q0, r0
    df["SoH_Q"] = df.QD_Ah / q0
    df["SoH_Q_nominal"] = df.QD_Ah / NOMINAL_AH
    df["R_ratio"] = df.IR_ohm / r0
    df["SoH_R_proxy"] = r0 / df.IR_ohm
    df["life_frac"] = df.cycle / df.cycle_life
    return df


def build_table():
    cells, split, counts = load_clean()
    df = pd.concat([cell_table(k, c) for k, c in cells.items()], ignore_index=True)
    df["split"] = df.cell_id.map(split)
    return df, counts


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    df, counts = build_table()
    print("cell counts after cleaning (b1, b2, b3):", counts, " total:", df.cell_id.nunique())
    print("rows total:", len(df))
    flags = [c for c in df.columns if c.startswith("flag_")]
    print("rows flagged per rule:", {f: int(df[f].sum()) for f in flags})
    print("rows dropped (any flag):", int((~df.valid).sum()))
    print("cells with IR zero rows:", sorted(df[df.flag_ir_zero].cell_id.unique()))
    clean = df[df.valid].drop(columns=flags + ["valid"])
    print("clean rows:", len(clean))
    print("IR_ohm after cleaning: min %.5f median %.5f max %.5f, NaN %d, zero %d" % (
        clean.IR_ohm.min(), clean.IR_ohm.median(), clean.IR_ohm.max(),
        clean.IR_ohm.isna().sum(), (clean.IR_ohm == 0).sum()))
    print("Q0 range %.4f to %.4f Ah, R0 range %.5f to %.5f ohm" % (
        clean.Q0_Ah.min(), clean.Q0_Ah.max(), clean.R0_ohm.min(), clean.R0_ohm.max()))
    print("SoH_Q range %.3f to %.3f; R_ratio range %.3f to %.3f" % (
        clean.SoH_Q.min(), clean.SoH_Q.max(), clean.R_ratio.min(), clean.R_ratio.max()))
    last = clean.groupby("cell_id").last()
    print("SoH_Q at last valid cycle: min %.3f median %.3f max %.3f" % (
        last.SoH_Q.min(), last.SoH_Q.median(), last.SoH_Q.max()))
    print("R_ratio at last valid cycle: min %.3f median %.3f max %.3f" % (
        last.R_ratio.min(), last.R_ratio.median(), last.R_ratio.max()))
    print("rows per SoH_Q bin:", pd.cut(clean.SoH_Q, [0, 0.8, 0.9, 1.0, 2]).value_counts().sort_index().to_dict())
    print("cells per split:", clean.groupby("split").cell_id.nunique().to_dict())
    df.to_csv(OUT / "cycle_table_with_flags.csv", index=False)
    clean.to_csv(OUT / "cycle_table.csv", index=False)
    print("saved outputs/cycle_table.csv and outputs/cycle_table_with_flags.csv")
