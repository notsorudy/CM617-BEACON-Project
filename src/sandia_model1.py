"""
BEACON: Model 1 on the Sandia data. Predict SoH from operating conditions and usage only.

Inputs (no capacity, energy, voltage, efficiency or measured Ah, so no target leakage):
  chemistry (one-hot LFP / NMC / NCA), temp_C, dod (%), c_discharge (C),
  efc_nominal = cycle_index x dod / 100 (charge cycled according to the test plan), days since start.
Target: SoH at each capacity check. The first check of each cell (SoH = 1 by definition) is not used.

Tests (all scores are out-of-fold, on data the model did not see):
  cells:      GroupKFold(5) with groups = cell
  conditions: LeaveOneGroupOut with groups = condition (chemistry, temperature, DoD, rate): 33 folds,
              every replicate of the held-out condition is unseen
Light tuning inside each training fold (GroupKFold(3) by cell).
Baselines: training mean; per-chemistry straight line in efc_nominal (fit on the training fold).

Outputs: outputs/sandia_m1_metrics.csv, outputs/sandia_m1_by_group.csv, outputs/sandia_m1_predictions.csv,
outputs/models/sandia_m1__*.joblib (trained on all data), figures/sandia_m1_*.png
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "outputs", ROOT / "figures"
(OUT / "models").mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
pd.set_option("display.width", 250)
SEED = 0
FEATS = ["chem_LFP", "chem_NMC", "chem_NCA", "temp_C", "dod", "c_discharge", "efc_nominal", "days"]
ALGOS = {
    "linear": (make_pipeline(StandardScaler(), LinearRegression()), {}),
    "tree": (DecisionTreeRegressor(random_state=SEED), {"max_depth": [3, 5, 8], "min_samples_leaf": [5, 15, 30]}),
    "forest": (RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1),
               {"max_depth": [6, 12, None], "min_samples_leaf": [2, 5, 10], "max_features": [0.5, 1.0]}),
    "gboost": (HistGradientBoostingRegressor(random_state=SEED, max_iter=300),
               {"learning_rate": [0.05, 0.1], "max_depth": [3, None], "min_samples_leaf": [5, 15]}),
}
BINS = [("SoH >= 0.9", 0.9, 9), ("0.8-0.9", 0.8, 0.9), ("< 0.8", 0, 0.8)]


def load():
    P = pd.read_csv(OUT / "sandia_soh_points.csv")
    first = P.groupby("cell_id").cycle_index.transform("min")
    P = P[P.cycle_index > first].copy()
    for c in ["LFP", "NMC", "NCA"]:
        P[f"chem_{c}"] = (P.chem == c).astype(int)
    P["efc_nominal"] = P.cycle_index * P.dod / 100
    P["condition"] = (P.chem + "_" + P.temp_C.astype(int).astype(str) + "C_" + P.dod.astype(int).astype(str)
                      + "_" + P.c_discharge.astype(str) + "C")
    return P.reset_index(drop=True)


def chem_line_baseline(tr, te):
    p = np.full(len(te), tr.SoH.mean())
    for c in ["LFP", "NMC", "NCA"]:
        a, b = tr.chem == c, (te.chem == c).to_numpy()
        if a.sum() >= 2 and b.any():
            k = np.polyfit(tr.loc[a, "efc_nominal"], tr.loc[a, "SoH"], 1)
            p[b] = np.polyval(k, te.loc[b, "efc_nominal"])
    return p


def run_split(P, split_name, outer):
    groups = P.cell_id if split_name == "cells" else P.condition
    preds = {k: np.full(len(P), np.nan) for k in list(ALGOS) + ["baseline_mean", "baseline_chem_line"]}
    for tr_i, te_i in outer.split(P, groups=groups):
        tr, te = P.iloc[tr_i], P.iloc[te_i]
        preds["baseline_mean"][te_i] = tr.SoH.mean()
        preds["baseline_chem_line"][te_i] = chem_line_baseline(tr, te)
        for name, (est, grid) in ALGOS.items():
            if grid:
                gs = GridSearchCV(est, grid, cv=GroupKFold(3), scoring="neg_root_mean_squared_error", n_jobs=-1)
                gs.fit(tr[FEATS], tr.SoH, groups=tr.cell_id)
                m = gs.best_estimator_
            else:
                m = clone(est).fit(tr[FEATS], tr.SoH)
            preds[name][te_i] = m.predict(te[FEATS])
    return preds


def metrics(y, p):
    return dict(n=len(y), R2=r2_score(y, p), RMSE=float(np.sqrt(mean_squared_error(y, p))), MAE=mean_absolute_error(y, p))


def main():
    P = load()
    print(f"points {len(P)} (first check of each cell removed), cells {P.cell_id.nunique()}, "
          f"conditions {P.condition.nunique()}")
    print("points per chemistry:", P.chem.value_counts().to_dict())
    rows, grp_rows = [], []
    out = P[["cell_id", "condition", "chem", "temp_C", "dod", "c_discharge", "efc_nominal", "days", "SoH"]].copy()
    for split_name, outer in [("cells", GroupKFold(5)), ("conditions", LeaveOneGroupOut())]:
        preds = run_split(P, split_name, outer)
        for algo, p in preds.items():
            out[f"{split_name}__{algo}"] = p
            rows.append(dict(split=split_name, algo=algo, subset="all", **metrics(P.SoH, p)))
            for c in ["LFP", "NMC", "NCA"]:
                m = (P.chem == c).to_numpy()
                rows.append(dict(split=split_name, algo=algo, subset=c, **metrics(P.SoH[m], p[m])))
            for lab, lo, hi in BINS:
                m = ((P.SoH >= lo) & (P.SoH < hi)).to_numpy()
                if m.sum():
                    rows.append(dict(split=split_name, algo=algo, subset=lab, **metrics(P.SoH[m], p[m])))
            e = pd.DataFrame({"g": P.condition, "err": np.abs(p - P.SoH)})
            for g, x in e.groupby("g"):
                grp_rows.append(dict(split=split_name, algo=algo, condition=g, MAE=x.err.mean(), n=len(x)))
        print(f"done: {split_name}", flush=True)
    M = pd.DataFrame(rows)
    M.to_csv(OUT / "sandia_m1_metrics.csv", index=False)
    pd.DataFrame(grp_rows).to_csv(OUT / "sandia_m1_by_group.csv", index=False)
    out.to_csv(OUT / "sandia_m1_predictions.csv", index=False)

    print("\nout-of-fold results, all points:")
    print(M[M.subset == "all"].pivot_table(index="algo", columns="split", values=["R2", "RMSE", "MAE"])
          .round(4).to_string())
    print("\nby chemistry and SoH range, RMSE:")
    print(M[M.subset != "all"].pivot_table(index=["split", "algo"], columns="subset", values="RMSE")
          .round(4).to_string())
    G = pd.DataFrame(grp_rows)
    w = G[(G.split == "conditions") & (G.algo == "forest")].sort_values("MAE", ascending=False)
    print("\nleave-condition-out, forest: worst 6 conditions by MAE:")
    print(w.head(6).round(4).to_string(index=False))

    # final models on all data, tuned by GroupKFold over cells
    for name, (est, grid) in ALGOS.items():
        if grid:
            gs = GridSearchCV(est, grid, cv=GroupKFold(5), scoring="neg_root_mean_squared_error", n_jobs=-1)
            gs.fit(P[FEATS], P.SoH, groups=P.cell_id)
            m, prm = gs.best_estimator_, gs.best_params_
        else:
            m, prm = clone(est).fit(P[FEATS], P.SoH), {}
        joblib.dump(m, OUT / "models" / f"sandia_m1__{name}.joblib", compress=3)
        print(f"final {name}: {prm}")

    figures(P, out)


def figures(P, out):
    cols = {"LFP": "tab:blue", "NMC": "tab:orange", "NCA": "tab:green"}
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for a, (split, algo, title) in zip(ax, [("cells", "forest", "held-out CELLS, random forest"),
                                            ("conditions", "forest", "held-out CONDITIONS, random forest"),
                                            ("conditions", "baseline_chem_line", "held-out CONDITIONS, baseline")]):
        p = out[f"{split}__{algo}"]
        for c, col in cols.items():
            m = out.chem == c
            a.scatter(out.SoH[m], p[m], s=10, alpha=0.6, color=col, label=c)
        a.plot([0.5, 1.02], [0.5, 1.02], "k--", lw=1)
        a.set(xlim=(0.5, 1.02), ylim=(0.5, 1.02), xlabel="actual SoH", ylabel="predicted SoH",
              title=f"{title}\nR2 = {r2_score(out.SoH, p):.3f}, RMSE = {np.sqrt(mean_squared_error(out.SoH, p)):.4f}")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "sandia_m1_pred_vs_actual.png", dpi=150)
    plt.close(fig)

    # example trajectories: held-out condition predictions vs actual
    ex = ["LFP_25C_100_3.0C", "LFP_35C_100_1.0C", "NMC_25C_20_0.5C", "NMC_35C_100_1.0C", "NCA_25C_60_0.5C",
          "NCA_25C_100_2.0C"]
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    for a, cond in zip(ax.flat, ex):
        g = out[out.condition == cond]
        for i, (cid, x) in enumerate(g.groupby("cell_id")):
            a.plot(x.efc_nominal, x.SoH, "o-", color="k", ms=3, lw=1, label="actual" if i == 0 else None)
            a.plot(x.efc_nominal, x["conditions__forest"], "s--", color="tab:red", ms=3, lw=1,
                   label="forest (condition unseen)" if i == 0 else None)
            a.plot(x.efc_nominal, x["conditions__baseline_chem_line"], ":", color="tab:blue", lw=1,
                   label="baseline (chem line)" if i == 0 else None)
        a.set(title=cond, xlabel="nominal equivalent full cycles", ylabel="SoH")
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "sandia_m1_trajectories.png", dpi=150)
    plt.close(fig)

    # permutation importance of the final forest, on all data (indicative only)
    m = joblib.load(OUT / "models" / "sandia_m1__forest.joblib")
    r = permutation_importance(m, P[FEATS], P.SoH, n_repeats=10, random_state=0, n_jobs=-1,
                               scoring="neg_root_mean_squared_error")
    imp = pd.Series(r.importances_mean, index=FEATS).sort_values()
    imp.to_csv(OUT / "sandia_m1_importance.csv")
    print("\npermutation importance (final forest, increase in RMSE):")
    print(imp.round(4).to_string())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(imp.index, imp.values)
    ax.set(xlabel="increase in RMSE when shuffled", title="Sandia Model 1 (random forest): input importance")
    fig.tight_layout()
    fig.savefig(FIG / "sandia_m1_importance.png", dpi=150)
    plt.close(fig)
    print("saved figures")


if __name__ == "__main__":
    main()
