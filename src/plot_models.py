"""BEACON: figures for Model 1 and Model 2 (predicted vs actual, error vs SoH,
feature importances, example trajectories)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score

from models import FEATURES, TARGET, load_table

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "outputs", ROOT / "figures"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
SPLITS = [("train", "TRAIN cells (41)", "tab:blue"), ("test", "TEST cells (43)", "tab:red"),
          ("test2_batch3", "TEST batch3 (40)", "tab:purple")]

pr = pd.read_csv(OUT / "predictions.csv")


def pred_vs_actual(col, target, label, fname, lims):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    for a, (split, title, c) in zip(ax, SPLITS):
        d = pr[pr.split == split]
        y, p = d[target], d[col]
        a.hexbin(y, p, gridsize=60, bins="log", cmap="Blues" if split == "train" else "Reds", mincnt=1,
                 extent=lims + lims)
        a.plot(lims, lims, "k--", lw=1)
        a.set(xlim=lims, ylim=lims, xlabel=f"actual {label}", ylabel=f"predicted {label}")
        a.set_title(f"{title}\nR2 = {r2_score(y, p):.3f}, RMSE = {np.sqrt(mean_squared_error(y, p)):.4f}",
                    color=c)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)


pred_vs_actual("M1_full__forest", "SoH_Q", "SoH_Q", "m1_full_pred_vs_actual.png", [0.75, 1.01])
pred_vs_actual("M1_nomeas__forest", "SoH_Q", "SoH_Q", "m1_nomeas_pred_vs_actual.png", [0.75, 1.01])
pred_vs_actual("M2__forest", "R_ratio", "R/R0", "m2_pred_vs_actual.png", [0.9, 1.42])
pred_vs_actual("M2__linear_formula", "R_ratio", "R/R0", "m2_formula_pred_vs_actual.png", [0.9, 1.42])

# error vs SoH (held-out cells: test + batch3)
ho = pr[pr.split != "train"].copy()
bins = np.arange(0.76, 1.02, 0.02)
ho["bin"] = pd.cut(ho.SoH_Q, bins)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for col, lab in [("M1_full__forest", "M1 full, forest"), ("M1_full__gboost", "M1 full, gboost"),
                 ("M1_full__linear", "M1 full, linear"), ("M1_nomeas__forest", "M1 no current meas., forest"),
                 ("M1_full__baseline_per_cycle", "baseline per-cycle mean")]:
    e = (ho[col] - ho.SoH_Q).abs().groupby(ho.bin, observed=True).mean()
    ax[0].plot([b.mid for b in e.index], e.values, "o-", ms=4, label=lab)
for col, lab in [("M2__forest", "M2 forest (SoH, cycle, T)"), ("M2_soh__forest", "forest, SoH only"),
                 ("M2__linear", "M2 linear regression"), ("M2__linear_formula", "linear formula 1+alpha(1-SoH)"),
                 ("M2__baseline_per_cycle", "baseline per-cycle mean")]:
    e = (ho[col] - ho.R_ratio).abs().groupby(ho.bin, observed=True).mean()
    ax[1].plot([b.mid for b in e.index], e.values, "o-", ms=4, label=lab)
ax[0].set(xlabel="actual SoH_Q", ylabel="mean absolute error in SoH_Q", title="Model 1 error vs SoH (held-out cells)")
ax[1].set(xlabel="actual SoH_Q", ylabel="mean absolute error in R/R0", title="Model 2 error vs SoH (held-out cells)")
for a in ax:
    a.invert_xaxis()
    a.legend(fontsize=8)
    a.axvspan(0.76, 0.8, color="grey", alpha=0.15)
fig.tight_layout()
fig.savefig(FIG / "error_vs_soh.png", dpi=150)
plt.close(fig)

# permutation importance on the test cells (more honest than impurity importance)
df = load_table()
te = df[df.split == "test"]
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
imp_rows = []
for a, model in zip(ax, ["M1_full", "M1_nomeas", "M2"]):
    est = joblib.load(OUT / "models" / f"{model}__forest.joblib")
    feats = FEATURES[model]
    r = permutation_importance(est, te[feats], te[TARGET[model]], n_repeats=5, random_state=0, n_jobs=-1,
                               scoring="neg_root_mean_squared_error")
    imp = pd.DataFrame({"feature": feats, "perm_mean": r.importances_mean, "perm_std": r.importances_std,
                        "impurity": est.feature_importances_}).sort_values("perm_mean")
    imp_rows.append(imp.assign(model=model))
    a.barh(imp.feature, imp.perm_mean, xerr=imp.perm_std, color="tab:blue")
    a.set(xlabel="increase in test RMSE when shuffled", title=f"{model} random forest")
fig.tight_layout()
fig.savefig(FIG / "feature_importance.png", dpi=150)
plt.close(fig)
pd.concat(imp_rows).to_csv(OUT / "feature_importance.csv", index=False)

# example trajectories for a few held-out cells
cells = ["b1c0", "b2c2", "b2c47", "b3c6", "b3c38"]
fig, ax = plt.subplots(2, len(cells), figsize=(4 * len(cells), 7.5), sharey="row")
for j, k in enumerate(cells):
    d = pr[pr.cell_id == k]
    ax[0, j].plot(d.cycle, d.SoH_Q, "k", lw=1.5, label="actual")
    ax[0, j].plot(d.cycle, d.M1_full__forest, "tab:red", lw=1, label="M1 full")
    ax[0, j].plot(d.cycle, d.M1_nomeas__forest, "tab:blue", lw=1, label="M1 no current meas.")
    ax[0, j].set_title(f"{k} ({d.split.iloc[0]})")
    ax[1, j].plot(d.cycle, d.R_ratio, "k", lw=1, label="actual")
    ax[1, j].plot(d.cycle, d.M2__forest, "tab:green", lw=1, label="M2 forest")
    ax[1, j].plot(d.cycle, d.M2__linear_formula, "tab:red", lw=1, label="linear formula")
    ax[1, j].set_xlabel("Cycle number")
ax[0, 0].set_ylabel("SoH_Q")
ax[1, 0].set_ylabel("R / R0")
ax[0, 0].legend(fontsize=8)
ax[1, 0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "example_trajectories.png", dpi=150)
print("saved figures")
print(pd.concat(imp_rows)[["model", "feature", "perm_mean", "impurity"]].round(4).to_string())
