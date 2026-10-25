"""
BEACON: convert one Severson .mat batch file into pickles.
Based on BuildPkl_Batch1/2/3.ipynb from the Severson repo. The three notebooks are
identical except for the file name and the key prefix, so this is one script.

Changes from the notebooks:
- h5py 3 removed Dataset.value, so we use ds[()] instead.
- Paths are relative to the project folder, not ./Data/.
- Mode "summary" only reads the summary fields, cycle_life and policy. It never
  touches the within-cycle data, so it uses very little memory.
- Mode "full" also reads the within-cycle data (same output as the notebooks).
  It stops early if the process uses more than MAX_RSS_GB of RAM.

Usage:
  python src/build_pkl.py 1 summary
  python src/build_pkl.py 1 full
"""
import gc
import os
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MAT_FILES = {
    1: "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    2: "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    3: "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
MAX_RSS_GB = 11.0

SUMMARY_FIELDS = {"IR": "IR", "QC": "QCharge", "QD": "QDischarge", "Tavg": "Tavg",
                  "Tmin": "Tmin", "Tmax": "Tmax", "chargetime": "chargetime", "cycle": "cycle"}
CYCLE_FIELDS = {"I": "I", "Qc": "Qc", "Qd": "Qd", "Qdlin": "Qdlin", "T": "T",
                "Tdlin": "Tdlin", "V": "V", "dQdV": "discharge_dQdV", "t": "t"}


def rss_gb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e9
    except ImportError:
        return 0.0


def read_cell_summary(f, batch, i):
    cl = f[batch["cycle_life"][i, 0]][()]
    policy = f[batch["policy_readable"][i, 0]][()].tobytes()[::2].decode()
    s = f[batch["summary"][i, 0]]
    summary = {k: np.hstack(s[mk][0, :].tolist()) for k, mk in SUMMARY_FIELDS.items()}
    return {"cycle_life": cl, "charge_policy": policy, "summary": summary}


def read_cell_cycles(f, batch, i):
    cycles = f[batch["cycles"][i, 0]]
    cycle_dict = {}
    for j in range(cycles["I"].shape[0]):
        cycle_dict[str(j)] = {k: np.hstack(f[cycles[mk][j, 0]][()]) for k, mk in CYCLE_FIELDS.items()}
    return cycle_dict


def main(batch_no, mode):
    mat = DATA / MAT_FILES[batch_no]
    prefix = f"b{batch_no}c"
    bat_dict = {}
    with h5py.File(mat, "r") as f:
        batch = f["batch"]
        num_cells = batch["summary"].shape[0]
        print(f"batch{batch_no}: {num_cells} cells in {mat.name}", flush=True)
        for i in range(num_cells):
            cell = read_cell_summary(f, batch, i)
            if mode == "full":
                cell["cycles"] = read_cell_cycles(f, batch, i)
                mem = rss_gb()
                print(f"  {prefix}{i}: {len(cell['cycles'])} cycles, RSS {mem:.2f} GB", flush=True)
                if mem > MAX_RSS_GB:
                    print(f"STOP: RSS {mem:.2f} GB above {MAX_RSS_GB} GB, full pkl not written")
                    return 2
            bat_dict[prefix + str(i)] = cell

    out = DATA / (f"batch{batch_no}_summary.pkl" if mode == "summary" else f"batch{batch_no}.pkl")
    with open(out, "wb") as fp:
        pickle.dump(bat_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {out.name}: {len(bat_dict)} cells, {out.stat().st_size / 1e6:.1f} MB")
    del bat_dict
    gc.collect()
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "summary"))
