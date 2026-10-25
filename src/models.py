"""
BEACON: Model 1 (SoH_Q) and Model 2 (resistance) on the Severson summary data.

Model 1, SoH model. Target SoH_Q. No discharge capacity, no IR in the features.
  M1_full    : cycle, current-cycle T_avg/T_max/T_min/chargetime, policy, early-cycle features
  M1_nomeas  : cycle, policy, early-cycle features (no current-cycle measurements)
Model 2, resistance model. Target R/R0 (main) and IR in ohm.
  M2         : SoH_Q, cycle, T_avg/T_max/T_min
  M2_soh     : SoH_Q only (pure R vs SoH curve, for the linearity test)
Baselines: training mean, per-cycle mean, and for Model 2 the linear formula
  R/R0 = 1 + alpha*(1 - SoH_Q) with alpha fitted on the training cells.
Split by cell: Severson split (train 41, test 43, batch3 40). Tuning with GroupKFold(5)
on the training cells only.

Outputs: outputs/results_metrics.csv, outputs/cv_results.csv, outputs/best_params.csv,
outputs/predictions.csv, outputs/models/*.joblib
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
(OUT / "models").mkdir(parents=True, exist_ok=True)
SEED = 0
EARLY = (2, 20)
MIN_CYCLE = 21

POLICY = ["C1", "Q1", "C2", "t_0_80_min", "Cavg_0_80"]
EARLY_FEATS = ["e_T_avg", "e_T_max", "e_T_min", "e_chargetime"]
CURRENT = ["T_avg", "T_max", "T_min", "chargetime_min"]
FEATURES = {
    "M1_full": ["cycle"] + CURRENT + POLICY + EARLY_FEATS,
    "M1_nomeas": ["cycle"] + POLICY + EARLY_FEATS,
    "M2": ["SoH_Q", "cycle", "T_avg", "T_max", "T_min"],
    "M2_soh": ["SoH_Q"],
}
TARGET = {"M1_full": "SoH_Q", "M1_nomeas": "SoH_Q", "M2": "R_ratio", "M2_soh": "R_ratio", "M2_IR": "IR_ohm"}
FEATURES["M2_IR"] = FEATURES["M2"]

ALGOS = {
    "linear": (make_pipeline(StandardScaler(), LinearRegression()), {}),
    "tree": (DecisionTreeRegressor(random_state=SEED),
             {"max_depth": [4, 6, 8, 12], "min_samples_leaf": [20, 100, 400]}),
    "forest": (RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=1),
               {"max_depth": [8, 14, 20], "min_samples_leaf": [5, 20, 80], "max_features": [0.5, 1.0]}),
    "gboost": (HistGradientBoostingRegressor(random_state=SEED, max_iter=300),
               {"learning_rate": [0.05, 0.1], "max_depth": [3, 6, None], "min_samples_leaf": [20, 100]}),
}
SOH_BINS = [(">=0.9", 0.9, 9.0), ("0.8-0.9", 0.8, 0.9), ("<0.8", 0.0, 0.8)]


def load_table():
    df = pd.read_csv(OUT / "cycle_table.csv")
    # early features only use T and chargetime, so only drop rows where those are bad.
    # (b1c18 cycles 2-53 are dropped from the main table for stuck IR, but T is fine there.)
    raw = pd.read_csv(OUT / "cycle_table_with_flags.csv")
    raw = raw[~raw.flag_zero_row & ~raw.flag_pause & raw.cycle.between(*EARLY)]
    early = raw.groupby("cell_id")[["T_avg", "T_max", "T_min", "chargetime_min"]].mean()
    early.columns = EARLY_FEATS
    df = df.join(early, on="cell_id")
    return df[df.cycle >= MIN_CYCLE].reset_index(drop=True)


def metrics(y, p):
    return dict(n_rows=len(y), R2=r2_score(y, p) if len(y) > 1 else np.nan,
                RMSE=float(np.sqrt(mean_squared_error(y, p))), MAE=mean_absolute_error(y, p))


def per_cycle_mean(train, target):
    """Mean target over training cells at each cycle number (21-cycle rolling mean), held flat
    beyond the last training cycle."""
    m = train.groupby("cycle")[target].mean()
    m = m.reindex(np.arange(0, int(train.cycle.max()) + 1)).interpolate().bfill()
    m = m.rolling(21, center=True, min_periods=1).mean()
    return lambda d: m.reindex(d.cycle.clip(upper=m.index.max()).astype(int)).to_numpy()


def fit_alpha(train):
    """Least squares alpha for R/R0 - 1 = alpha * (1 - SoH_Q) (line through R/R0 = 1 at SoH = 1)."""
    x = 1 - train.SoH_Q.to_numpy()
    y = train.R_ratio.to_numpy() - 1
    return float(np.sum(x * y) / np.sum(x * x))


def evaluate(df, pred, model, algo, target, rows):
    """Overall metrics per split, error by SoH bin, and near end of life."""
    for split in ["train", "test", "test2_batch3"]:
        d = df[df.split == split]
        y, p = d[target].to_numpy(), pred[d.index]
        rows.append(dict(model=model, algo=algo, target=target, split=split, subset="all",
                         n_cells=d.cell_id.nunique(), **metrics(y, p)))
        for name, lo, hi in SOH_BINS:
            m = ((d.SoH_Q >= lo) & (d.SoH_Q < hi)).to_numpy()
            if m.sum() > 0:
                rows.append(dict(model=model, algo=algo, target=target, split=split, subset=f"SoH {name}",
                                 n_cells=d[m].cell_id.nunique(), **metrics(y[m], p[m])))
        m = (d.life_frac >= 0.9).to_numpy()
        rows.append(dict(model=model, algo=algo, target=target, split=split, subset="last 10% of life",
                         n_cells=d[m].cell_id.nunique(), **metrics(y[m], p[m])))


def main():
    t0 = time.time()
    df = load_table()
    train = df[df.split == "train"]
    groups = train.cell_id.to_numpy()
    cv = GroupKFold(n_splits=5)
    print(f"rows used (cycle >= {MIN_CYCLE}): {len(df)}; train rows {len(train)} from {train.cell_id.nunique()} cells")
    print("corr(SoH_Q, chargetime_min) on train:", round(train.SoH_Q.corr(train.chargetime_min), 3),
          " corr(SoH_Q, T_avg):", round(train.SoH_Q.corr(train.T_avg), 3))

    rows, cv_rows, best_rows = [], [], []
    preds = df[["cell_id", "split", "cycle", "SoH_Q", "R_ratio", "IR_ohm", "R0_ohm", "life_frac"]].copy()

    # baselines
    for model in ["M1_full", "M2", "M2_IR"]:
        target = TARGET[model]
        p_mean = np.full(len(df), train[target].mean())
        evaluate(df, p_mean, model, "baseline_mean", target, rows)
        p_cyc = per_cycle_mean(train, target)(df)
        evaluate(df, p_cyc, model, "baseline_per_cycle", target, rows)
        preds[f"{model}__baseline_per_cycle"] = p_cyc
    alpha = fit_alpha(train)
    p_lin = 1 + alpha * (1 - df.SoH_Q.to_numpy())
    evaluate(df, p_lin, "M2", "linear_formula", "R_ratio", rows)
    evaluate(df, p_lin * df.R0_ohm.to_numpy(), "M2_IR", "linear_formula_x_R0", "IR_ohm", rows)
    preds["M2__linear_formula"] = p_lin
    print(f"alpha fitted on train cells: {alpha:.4f}")

    # ML models
    for model in ["M1_full", "M1_nomeas", "M2", "M2_soh", "M2_IR"]:
        feats, target = FEATURES[model], TARGET[model]
        X, y = train[feats], train[target]
        for algo, (est, grid) in ALGOS.items():
            if model == "M2_soh" and algo == "linear":
                continue  # same as a straight line with intercept, covered in linearity.py
            ts = time.time()
            if grid:
                gs = GridSearchCV(est, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
                gs.fit(X, y, groups=groups)
                best, params = gs.best_estimator_, gs.best_params_
            else:
                best, params = clone(est).fit(X, y), {}
            cvr = cross_validate(clone(best), X, y, groups=groups, cv=cv, n_jobs=-1,
                                 scoring=["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"])
            cv_rows.append(dict(model=model, algo=algo, target=target,
                                cv_R2_mean=cvr["test_r2"].mean(), cv_R2_std=cvr["test_r2"].std(),
                                cv_RMSE_mean=-cvr["test_neg_root_mean_squared_error"].mean(),
                                cv_RMSE_std=cvr["test_neg_root_mean_squared_error"].std(),
                                cv_MAE_mean=-cvr["test_neg_mean_absolute_error"].mean()))
            best_rows.append(dict(model=model, algo=algo, params=json.dumps(params)))
            p = best.predict(df[feats])
            evaluate(df, p, model, algo, target, rows)
            preds[f"{model}__{algo}"] = p
            if model == "M2":
                evaluate(df, p * df.R0_ohm.to_numpy(), "M2_IR_from_ratio", algo, "IR_ohm", rows)
            joblib.dump(best, OUT / "models" / f"{model}__{algo}.joblib", compress=3)
            r = [x for x in rows if x["model"] == model and x["algo"] == algo and x["subset"] == "all"]
            print(f"{model:9s} {algo:7s} {params} | CV R2 {cv_rows[-1]['cv_R2_mean']:.3f} | "
                  + " | ".join(f"{x['split']} R2 {x['R2']:.3f} RMSE {x['RMSE']:.4f}" for x in r)
                  + f" | {time.time() - ts:.0f}s", flush=True)

    # chained pipeline: Model 1 predicted SoH fed into Model 2 (test cells, SoH not measured)
    m2 = joblib.load(OUT / "models" / "M2__forest.joblib")
    for m1 in ["M1_full", "M1_nomeas"]:
        soh_hat = preds[f"{m1}__forest"].to_numpy()
        Xc = df[FEATURES["M2"]].copy()
        Xc["SoH_Q"] = soh_hat
        evaluate(df, m2.predict(Xc), "chain_" + m1 + "_to_M2", "forest", "R_ratio", rows)
        evaluate(df, 1 + alpha * (1 - soh_hat), "chain_" + m1 + "_to_M2", "linear_formula", "R_ratio", rows)

    pd.DataFrame(rows).to_csv(OUT / "results_metrics.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(OUT / "cv_results.csv", index=False)
    pd.DataFrame(best_rows).to_csv(OUT / "best_params.csv", index=False)
    preds.to_csv(OUT / "predictions.csv", index=False)
    json.dump({"alpha_train": alpha}, open(OUT / "alpha.json", "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
