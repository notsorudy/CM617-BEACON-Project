"""BEACON: leakage check for Sandia Model 1. Same tests, without the 'days' input.
Why: at a fixed current a cell with less capacity finishes each cycle sooner, so elapsed days after N
cycles partly encodes the remaining capacity. Output: outputs/sandia_m1_metrics_nodays.csv"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

import sandia_model1 as m1

m1.FEATS = [f for f in m1.FEATS if f != "days"]
P = m1.load()
rows = []
out = P[["cell_id", "condition", "chem", "efc_nominal", "SoH"]].copy()
for split_name, outer in [("cells", GroupKFold(5)), ("conditions", LeaveOneGroupOut())]:
    preds = m1.run_split(P, split_name, outer)
    for algo, p in preds.items():
        out[f"{split_name}__{algo}"] = p
        rows.append(dict(split=split_name, algo=algo, subset="all", **m1.metrics(P.SoH, p)))
        for c in ["LFP", "NMC", "NCA"]:
            m = (P.chem == c).to_numpy()
            rows.append(dict(split=split_name, algo=algo, subset=c, **m1.metrics(P.SoH[m], p[m])))
M = pd.DataFrame(rows)
M.to_csv(m1.OUT / "sandia_m1_metrics_nodays.csv", index=False)
out.to_csv(m1.OUT / "sandia_m1_predictions_nodays.csv", index=False)
W = pd.read_csv(m1.OUT / "sandia_m1_metrics.csv")
both = pd.concat([W.assign(inputs="with days"), M.assign(inputs="without days")])
pd.set_option("display.width", 250)
print(both[(both.subset == "all") & both.algo.isin(["forest", "gboost", "tree", "linear"])]
      .pivot_table(index=["algo", "inputs"], columns="split", values=["R2", "RMSE"]).round(4).to_string())
print(both[(both.algo == "forest") & both.subset.isin(["LFP", "NMC", "NCA"])]
      .pivot_table(index=["subset", "inputs"], columns="split", values="RMSE").round(4).to_string())

# final models without 'days' (the main Sandia Model 1), trained on all points
import joblib
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

for name, (est, grid) in m1.ALGOS.items():
    if grid:
        gs = GridSearchCV(est, grid, cv=GroupKFold(5), scoring="neg_root_mean_squared_error", n_jobs=-1)
        gs.fit(P[m1.FEATS], P.SoH, groups=P.cell_id)
        model, prm = gs.best_estimator_, gs.best_params_
    else:
        model, prm = clone(est).fit(P[m1.FEATS], P.SoH), {}
    joblib.dump(model, m1.OUT / "models" / f"sandia_m1_nodays__{name}.joblib", compress=3)
    print(f"final (no days) {name}: {prm}")
print("inputs:", m1.FEATS)
