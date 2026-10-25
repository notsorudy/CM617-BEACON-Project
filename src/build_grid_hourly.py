"""
BEACON: convert the Grid-India SCADA files (Mendeley dataset 10.17632/y58jknpgs8.2) into one hourly table.

Three formats in the download:
- Sep 2021 to Sep 2022: 5-minute rows, Sheet1, generation by source + demand + inertia tags
- Oct 2022 to Dec 2023: 10-second rows, same columns
- Jan 2024 to Jun 2025: hourly rows, sheet "Report", demand / wind / solar / total only
Every file is reduced to hourly means (timestamp = start of the hour, as given in the files).
Column names are taken from the second header row of Sheet1 (the SCADA tag names), so a file
with a different layout is reported, not silently misread.

Output: outputs/grid_india_hourly.csv, outputs/grid_india_files.csv (per-file log)
"""
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data" / "demand" / "Electricity Demand, Solar and Wind Generation Data"
OUT = ROOT / "outputs"

TAGS = {"TOTAL_THM_ONLY|P": "thermal_MW", "TOTAL_NLDC_HYD|P": "hydro_MW", "TOTAL_NLDC_GAS|P": "gas_MW",
        "TOTAL_NLDC_NPC|P": "nuclear_MW", "ALL_INDIA_WIND|P": "wind_MW", "ALL_IND_SOLAR|P": "solar_MW",
        "NLDC_DEMAND|P": "demand_MW", "INTRA_SYSTEM_INTERTIA|P": "tag_INTRA_SYSTEM_INTERTIA",
        "ER_Total_Inertia|P": "tag_ER_Total_Inertia", "SYS_IN|P": "tag_SYS_IN"}
REPORT = {"Demand (MW)": "demand_MW", "Wind (MW)": "wind_MW", "Solar (MW)": "solar_MW",
          "Total Generation (MW)": "total_gen_MW"}


def read_file(path):
    xl = pd.ExcelFile(path, engine="calamine")
    if "Report" in xl.sheet_names:
        df = xl.parse("Report")
        df["time"] = pd.to_datetime(df.Timestamp, format="%d-%m-%Y %H:%M:%S", errors="coerce")
        dropped = df.loc[df.time.isna(), "Timestamp"].astype(str).tolist()
        df = df.rename(columns=REPORT)[["time"] + list(REPORT.values())]
        for c in REPORT.values():
            df[c] = pd.to_numeric(df[c], errors="coerce")
        fmt = "hourly report"
    else:
        raw = xl.parse("Sheet1", header=None)
        names = raw.iloc[1].tolist()
        cols = {}
        for i, n in enumerate(names):
            if isinstance(n, str) and n in TAGS:
                cols[TAGS[n]] = i
        missing = [v for v in TAGS.values() if v not in cols]
        df = pd.DataFrame({"time": pd.to_datetime(raw.iloc[2:, 0], errors="coerce")})
        dropped = raw.iloc[2:, 0][df.time.isna().to_numpy()].astype(str).tolist()
        for k, i in cols.items():
            df[k] = pd.to_numeric(raw.iloc[2:, i], errors="coerce").to_numpy()
        fmt = "SCADA sheet1" + (f" MISSING {missing}" if missing else "")
    df = df.dropna(subset=["time"])
    step = df.time.diff().median()
    df["hour"] = df.time.dt.floor("h")
    num = [c for c in df.columns if c not in ("time", "hour")]
    hourly = df.groupby("hour")[num].mean()
    hourly["n_samples"] = df.groupby("hour").size()
    info = dict(file=path.name, format=fmt, rows=len(df), first=df.time.min(), last=df.time.max(),
                median_step_s=step.total_seconds(), hours=len(hourly), non_date_rows=len(dropped),
                non_date_labels="|".join(sorted(set(dropped)))[:80])
    return hourly.reset_index().rename(columns={"hour": "time"}), info


if __name__ == "__main__":
    files = sorted(DIR.glob("*.xlsx"))
    with ProcessPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(read_file, files))
    info = pd.DataFrame([r[1] for r in res]).sort_values("first")
    pd.set_option("display.width", 250)
    print(info.to_string(index=False))
    info.to_csv(OUT / "grid_india_files.csv", index=False)
    h = pd.concat([r[0] for r in res], ignore_index=True).sort_values("time")
    dup = h.time.duplicated(keep=False)
    print(f"\nhourly rows {len(h)}, duplicated hours across files {int(dup.sum())}")
    h = h.groupby("time", as_index=False).mean(numeric_only=True)
    h.to_csv(OUT / "grid_india_hourly.csv", index=False)
    print(f"saved outputs/grid_india_hourly.csv: {len(h)} hours, {h.time.min()} to {h.time.max()}")
