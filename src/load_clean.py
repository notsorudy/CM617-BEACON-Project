"""
BEACON: load the summary pickles and apply the cleaning and merging rules from the
Severson repo notebook "Load Data.ipynb" (checked line by line against it).

- batch1: delete b1c8, b1c10, b1c12, b1c13, b1c22 (they do not reach 80% capacity).
- b2c7, b2c8, b2c9, b2c15, b2c16 are the continuation of b1c0..b1c4. Their summary data
  is appended to the batch1 cell (cycle numbers shifted by the batch1 length), cycle_life
  is increased by 662, 981, 1060, 208, 482, and the batch2 keys are deleted.
  (The notebook comment says "four cells", but the code moves five. We follow the code.)
- batch3: delete noisy cells b3c37, b3c2, b3c23, b3c32, b3c42, b3c43.
- Split (indices into the merged dict, in insertion order batch1, batch2, batch3):
  test = 0, 2, 4, ... of batch1+2 plus index 83; train = 1, 3, ..., 81;
  secondary test = all of batch3.
"""
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

B1_DELETE = ["b1c8", "b1c10", "b1c12", "b1c13", "b1c22"]
B2_KEYS = ["b2c7", "b2c8", "b2c9", "b2c15", "b2c16"]
B1_KEYS = ["b1c0", "b1c1", "b1c2", "b1c3", "b1c4"]
ADD_LEN = [662, 981, 1060, 208, 482]
B3_DELETE = ["b3c37", "b3c2", "b3c23", "b3c32", "b3c42", "b3c43"]


def load_summary(batch_no):
    with open(DATA / f"batch{batch_no}_summary.pkl", "rb") as f:
        return pickle.load(f)


def load_clean(verbose=False):
    b1, b2, b3 = load_summary(1), load_summary(2), load_summary(3)
    raw = (len(b1), len(b2), len(b3))
    for k in B1_DELETE:
        del b1[k]
    for i, bk in enumerate(B1_KEYS):
        b1[bk]["cycle_life"] = b1[bk]["cycle_life"] + ADD_LEN[i]
        s1, s2 = b1[bk]["summary"], b2[B2_KEYS[i]]["summary"]
        for j in s1:
            if j == "cycle":
                s1[j] = np.hstack((s1[j], s2[j] + len(s1[j])))
            else:
                s1[j] = np.hstack((s1[j], s2[j]))
    for k in B2_KEYS:
        del b2[k]
    for k in B3_DELETE:
        del b3[k]
    counts = (len(b1), len(b2), len(b3))
    if verbose:
        print("raw cell counts:", raw, " after cleaning:", counts)
    cells = {**b1, **b2, **b3}
    keys = list(cells)
    n12 = counts[0] + counts[1]
    split = {}
    test_ind = set(np.hstack((np.arange(0, n12, 2), 83)).tolist())
    train_ind = set(np.arange(1, n12 - 1, 2).tolist())
    for idx, k in enumerate(keys):
        if idx >= n12:
            split[k] = "test2_batch3"
        elif idx in test_ind:
            split[k] = "test"
        elif idx in train_ind:
            split[k] = "train"
        else:
            split[k] = "unused"
    return cells, split, counts


if __name__ == "__main__":
    cells, split, counts = load_clean(verbose=True)
    from collections import Counter
    print("split sizes:", Counter(split.values()))
    print("index 83 is", list(cells)[83])
