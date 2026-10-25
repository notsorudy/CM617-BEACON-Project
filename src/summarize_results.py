"""BEACON: compact results tables from outputs/results_metrics.csv and cv_results.csv."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
pd.set_option("display.width", 220)

r = pd.read_csv(OUT / "results_metrics.csv")
cv = pd.read_csv(OUT / "cv_results.csv")

main = r[r.subset == "all"].pivot_table(index=["model", "algo"], columns="split", values=["R2", "RMSE", "MAE"])
main = main.reindex(columns=["train", "test", "test2_batch3"], level=1)
cvs = cv.set_index(["model", "algo"])[["cv_R2_mean", "cv_R2_std", "cv_RMSE_mean"]]
cvs.columns = pd.MultiIndex.from_tuples([("GroupKFold", c) for c in cvs.columns])
summary = main.join(cvs)
summary.round(4).to_csv(OUT / "results_summary.csv")
print(summary.round(4).to_string())

print("\nError by SoH range and near end of life (test and batch3, main models):")
keep = [("M1_full", "forest"), ("M1_full", "gboost"), ("M1_full", "linear"), ("M1_full", "baseline_per_cycle"),
        ("M1_nomeas", "forest"), ("M2", "forest"), ("M2", "gboost"), ("M2", "linear"), ("M2", "linear_formula"),
        ("M2", "baseline_per_cycle"), ("M2_soh", "forest"), ("M2_IR_from_ratio", "forest"),
        ("M2_IR", "linear_formula_x_R0"), ("M2_IR", "forest")]
b = r[r.split.isin(["test", "test2_batch3"]) & (r.subset != "all")]
b = b[[(m, a) in keep for m, a in zip(b.model, b.algo)]]
bt = b.pivot_table(index=["model", "algo"], columns=["split", "subset"], values="MAE")
bt.round(4).to_csv(OUT / "results_by_soh_range.csv")
print(bt.round(4).to_string())
print("\nrows per bin:")
print(r[(r.model == "M2") & (r.algo == "forest")].pivot_table(index="subset", columns="split", values="n_rows"))
print("\nChained (Model 1 predicted SoH into Model 2):")
print(r[r.model.str.startswith("chain") & (r.subset == "all")][["model", "algo", "split", "R2", "RMSE", "MAE"]]
      .round(4).to_string(index=False))
