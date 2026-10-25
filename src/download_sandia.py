"""
BEACON: download the Sandia (SNL) per-cycle data for all 86 cells from batteryarchive.org.

Source: batteryarchive.org "Cycle Test Data" public dashboard (Redash). The dashboard's public link key
is used, exactly as the web page itself does when it draws the charts.
Study: Preger et al. 2020, J. Electrochem. Soc. (Sandia), commercial 18650 LFP / NMC / NCA cells.

Queries used (same ones the dashboard uses):
  26 "Energy and Capacity Decay"      -> ah_c, ah_d, e_c, e_d per cycle
  40 "Max and Min Voltage by Cycle"   -> v_max, v_min, v_c_mean, v_d_mean per cycle
  28 "Efficiencies"                   -> ah_eff, e_eff per cycle

Raw JSON responses are kept in data/sandia/cycle/raw/ (unchanged); one wide CSV per cell is written to
data/sandia/cycle/. Re-running skips cells that are already downloaded.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sandia" / "cycle"
RAW = OUT / "raw"
RAW.mkdir(parents=True, exist_ok=True)
BASE = "https://database.batteryarchive.org"
# public link key of the batteryarchive.org cell-data dashboard (shown in the dashboard's share link);
# set it in the environment before running: $env:BATTERYARCHIVE_KEY = "<key>"
KEY = os.environ.get("BATTERYARCHIVE_KEY")
if not KEY:
    sys.exit("Set BATTERYARCHIVE_KEY to the public dashboard key from batteryarchive.org (see README).")
QUERIES = {26: "capacity_energy", 40: "voltage", 28: "efficiency"}


def safe(cell_id):
    return re.sub(r"[^A-Za-z0-9_.-]", "-", cell_id)


def fetch(qid, cell_id, tries=4):
    body = {"id": qid, "parameters": {"cell_id": [cell_id]}, "apply_auto_limit": False, "max_age": -1}
    for t in range(tries):
        try:
            r = requests.post(f"{BASE}/api/queries/{qid}/results", params={"api_key": KEY}, json=body, timeout=120)
            r.raise_for_status()
            j = r.json()
            if "query_result" in j:
                return j
            time.sleep(3 + 3 * t)  # a job was queued, try again
        except requests.RequestException as e:
            print(f"   retry {t + 1} for q{qid} {cell_id}: {e}")
            time.sleep(3 + 3 * t)
    raise RuntimeError(f"failed q{qid} {cell_id}")


def to_wide(responses):
    """One row per cycle_index. test_time comes from the capacity query (the efficiency query has none)."""
    frames, times = [], []
    for j in responses:
        rows = pd.DataFrame(j["query_result"]["data"]["rows"])
        rows["var"] = rows.series.str.split(":").str[0].str.strip()
        rows["value"] = pd.to_numeric(rows.value, errors="coerce")
        frames.append(rows.pivot_table(index="cycle_index", columns="var", values="value", aggfunc="first"))
        if "test_time" in rows:
            times.append(rows.groupby("cycle_index").test_time.first())
    w = pd.concat(frames, axis=1)
    w.insert(0, "test_time", times[0].reindex(w.index))
    return w.reset_index().sort_values("cycle_index")


def main():
    ids = requests.get(f"{BASE}/api/queries/26/dropdowns/1", params={"api_key": KEY}, timeout=60).json()
    ids = [x["value"] if isinstance(x, dict) else x for x in (ids.get("rows", ids) if isinstance(ids, dict) else ids)]
    ids = sorted(x for x in ids if isinstance(x, str) and x.startswith("SNL_"))
    if "--reverse" in sys.argv:   # a second process can work from the other end of the list
        ids = ids[::-1]
    print(f"{len(ids)} SNL cells listed")
    log = []
    for n, cid in enumerate(ids, 1):
        csv = OUT / f"{safe(cid)}.csv"
        if csv.exists():
            continue
        resp = []
        for qid, name in QUERIES.items():
            f = RAW / f"{safe(cid)}__{name}.json"
            if f.exists():
                j = json.loads(f.read_text())
            else:
                j = fetch(qid, cid)
                f.write_text(json.dumps(j))
                time.sleep(0.5)
            resp.append(j)
        w = to_wide(resp)
        w.insert(0, "cell_id", cid)
        w.to_csv(csv, index=False)
        log.append(dict(cell_id=cid, cycles=len(w), first=w.cycle_index.min(), last=w.cycle_index.max(),
                        columns="|".join(c for c in w.columns if c not in ("cell_id", "cycle_index", "test_time"))))
        print(f"[{n:2d}/{len(ids)}] {cid}: {len(w)} cycles", flush=True)
    if log:
        pd.DataFrame(log).to_csv(OUT / "download_log.csv", mode="a", index=False,
                                 header=not (OUT / "download_log.csv").exists())
    print("done")


if __name__ == "__main__":
    main()
