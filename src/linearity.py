"""
BEACON: is R/R0 = 1 + alpha*(1 - SoH_Q) (Chen, Liu & Fu 2026) a good fit to the Severson data?

Fits on the training cells, evaluated on test cells and batch3:
  formula   : R/R0 = 1 + alpha*x,            x = 1 - SoH_Q   (the paper's form)
  linear    : R/R0 = a + b*x                 (free intercept)
  quadratic : R/R0 = a + b*x + c*x^2
  forest    : random forest on SoH_Q only (M2_soh) and on SoH_Q + cycle + T (M2), from models.py
Also alpha per cell, to see whether one alpha fits all cells.

Outputs: outputs/linearity_results.csv, outputs/alpha_per_cell.csv,
figures/linearity_R_vs_SoH.png, figures/linearity_residuals.png, figures/linearity_alpha_per_cell.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "outputs", ROOT / "figures"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
SPLIT_NAMES = {"train": "train (41 cells)", "test": "test (43 cells)", "test2_batch3": "batch3 test (40 cells)"}

pr = pd.read_csv(OUT / "predictions.csv")
tr = pr[pr.split == "train"]
x_tr, y_tr = 1 - tr.SoH_Q.to_numpy(), tr.R_ratio.to_numpy()

alpha = float(np.sum(x_tr * (y_tr - 1)) / np.sum(x_tr ** 2))
lin = np.polyfit(x_tr, y_tr, 1)
quad = np.polyfit(x_tr, y_tr, 2)
fits = {
    "formula 1+alpha*x": lambda d: 1 + alpha * (1 - d.SoH_Q.to_numpy()),
    "linear a+b*x": lambda d: np.polyval(lin, 1 - d.SoH_Q.to_numpy()),
    "quadratic": lambda d: np.polyval(quad, 1 - d.SoH_Q.to_numpy()),
    "forest, SoH only": lambda d: d["M2_soh__forest"].to_numpy(),
    "forest, SoH+cycle+T": lambda d: d["M2__forest"].to_numpy(),
}

rows = []
for name, f in fits.items():
    for split, d in pr.groupby("split"):
        y, p = d.R_ratio.to_numpy(), f(d)
        rows.append(dict(fit=name, split=split, n_rows=len(d), R2=r2_score(y, p),
                         RMSE=np.sqrt(mean_squared_error(y, p)), MAE=mean_absolute_error(y, p)))
res = pd.DataFrame(rows)
res.to_csv(OUT / "linearity_results.csv", index=False)
# coefficients, so Task 3 can use the exact fitted curves (x = 1 - SoH_Q)
json.dump({"alpha_formula": alpha, "linear_a_b": [lin[1], lin[0]],
           "quadratic_a_b_c": [quad[2], quad[1], quad[0]],
           "fit_range_SoH": [float(tr.SoH_Q.min()), float(tr.SoH_Q.max())],
           "data_range_SoH_all_cells": [float(pr.SoH_Q.min()), float(pr.SoH_Q.max())]},
          open(OUT / "resistance_curve_coefs.json", "w"), indent=1)
print(f"alpha (formula, train cells) = {alpha:.4f}")
print(f"linear a+b*x: a = {lin[1]:.4f}, b = {lin[0]:.4f}")
print(f"quadratic: a = {quad[2]:.4f}, b = {quad[1]:.4f}, c = {quad[0]:.4f}")
print(res.pivot(index="fit", columns="split", values=["R2", "RMSE"]).round(4).to_string())

# alpha per cell (line through 1), all cells
cells = []
for k, d in pr.groupby("cell_id"):
    x, y = 1 - d.SoH_Q.to_numpy(), d.R_ratio.to_numpy()
    a = float(np.sum(x * (y - 1)) / np.sum(x ** 2))
    p = 1 + a * x
    cells.append(dict(cell_id=k, split=d.split.iloc[0], batch=int(k[1]), alpha=a,
                      R2_own_alpha=r2_score(y, p), RMSE_own_alpha=np.sqrt(mean_squared_error(y, p)),
                      final_SoH=d.SoH_Q.iloc[-1], final_R_ratio=d.R_ratio.iloc[-1]))
ac = pd.DataFrame(cells)
ac.to_csv(OUT / "alpha_per_cell.csv", index=False)
print("\nalpha per cell by batch:")
print(ac.groupby("batch").alpha.describe().round(3).to_string())
print("alpha per cell, all: mean %.3f, std %.3f, CV %.2f; median R2 with own alpha %.3f" % (
    ac.alpha.mean(), ac.alpha.std(), ac.alpha.std() / ac.alpha.mean(), ac.R2_own_alpha.median()))
print("cells where own-alpha line has R2 < 0.5:", int((ac.R2_own_alpha < 0.5).sum()), "of", len(ac))

# residual structure of the formula, binned by SoH
bins = np.arange(0.76, 1.02, 0.02)
pr["SoH_bin"] = pd.cut(pr.SoH_Q, bins)
pr["res_formula"] = pr.R_ratio - fits["formula 1+alpha*x"](pr)
pr["res_forest"] = pr.R_ratio - pr["M2_soh__forest"]
binned = pr.groupby("SoH_bin", observed=True).agg(
    SoH=("SoH_Q", "mean"), R_mean=("R_ratio", "mean"), R_std=("R_ratio", "std"), n=("R_ratio", "size"),
    res_formula=("res_formula", "mean"), res_forest=("res_forest", "mean"))
print("\nbinned R/R0 and mean residuals (all cells):")
print(binned.round(4).to_string())

# figure 1: R/R0 vs SoH with fits
s = np.linspace(pr.SoH_Q.min(), 1.006, 200)
f_soh = joblib.load(OUT / "models" / "M2_soh__forest.joblib")
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for a, (split, d) in zip(ax, [("train", tr), ("test + batch3", pr[pr.split != "train"])]):
    a.hexbin(d.SoH_Q, d.R_ratio, gridsize=70, bins="log", cmap="Greys", mincnt=1)
    b = d.groupby(pd.cut(d.SoH_Q, bins), observed=True).agg(S=("SoH_Q", "mean"), R=("R_ratio", "mean"),
                                                            sd=("R_ratio", "std"))
    a.errorbar(b.S, b.R, yerr=b.sd, fmt="o", color="k", ms=4, capsize=2, label="binned mean +/- 1 std")
    a.plot(s, 1 + alpha * (1 - s), color="tab:red", lw=2, label=f"linear formula, alpha = {alpha:.2f}")
    a.plot(s, np.polyval(quad, 1 - s), color="tab:blue", lw=2, ls="--", label="quadratic")
    a.plot(s, f_soh.predict(pd.DataFrame({"SoH_Q": s})), color="tab:green", lw=2, label="random forest (SoH only)")
    a.set(xlabel="SoH_Q = QD / Q0", ylabel="R / R0", title=f"{split} cells", ylim=(0.9, 1.42))
    a.invert_xaxis()
ax[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Resistance vs capacity SoH (fits use training cells only)")
fig.tight_layout()
fig.savefig(FIG / "linearity_R_vs_SoH.png", dpi=150)

# figure 2: residuals of the formula
fig, ax = plt.subplots(figsize=(8, 4.5))
sub = pr.sample(min(20000, len(pr)), random_state=0)
ax.scatter(sub.SoH_Q, sub.res_formula, s=2, alpha=0.15, color="grey", label="rows (20k sample)")
ax.plot(binned.SoH, binned.res_formula, "o-", color="tab:red", label="binned mean residual, linear formula")
ax.plot(binned.SoH, binned.res_forest, "s-", color="tab:green", label="binned mean residual, forest (SoH only)")
ax.axhline(0, color="k", lw=1)
ax.set(xlabel="SoH_Q", ylabel="R/R0 actual - predicted", title="Residuals vs SoH (all 124 cells)", ylim=(-0.2, 0.3))
ax.invert_xaxis()
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "linearity_residuals.png", dpi=150)

# figure 3: alpha per cell
fig, ax = plt.subplots(figsize=(7, 4.5))
for b, c in [(1, "tab:blue"), (2, "tab:orange"), (3, "tab:green")]:
    ax.hist(ac[ac.batch == b].alpha, bins=np.linspace(ac.alpha.min(), ac.alpha.max(), 30), alpha=0.6,
            color=c, label=f"batch {b}")
ax.axvline(alpha, color="k", ls="--", label=f"pooled train alpha = {alpha:.2f}")
ax.set(xlabel="alpha fitted per cell", ylabel="number of cells", title="alpha is not the same for every cell")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "linearity_alpha_per_cell.png", dpi=150)
print("saved figures")
