# BEACON progress log

All numbers below come from scripts that were run. Scripts are in `src/`, figures in `figures/`, tables in `outputs/`.

## Setup (23 Oct 2026)

- Machine: 16 GB RAM, 20 logical cores, Windows 11. Python 3.11.9 venv in `.venv/`.
- Packages: h5py 3.16.0, numpy 2.4.6, pandas 3.1.0, scikit-learn 1.9.1, matplotlib 3.11.2, scipy 1.17.1, psutil. CPU only, no GPU or PyTorch.
- Data files in `data/`:
  - `2017-05-12_batchdata_updated_struct_errorcorrect.mat` (batch1, 3.03 GB)
  - `2017-06-30_batchdata_updated_struct_errorcorrect.mat` (batch2, 2.01 GB)
  - `2018-04-12_batchdata_updated_struct_errorcorrect.mat` (batch3, 3.24 GB)
- git is not installed and direct downloads from the shell are blocked on this machine, so the repo notebooks were read through the GitHub API in the browser. Their code is reproduced in the scripts below (not cloned into `repo/`).

## Task 1 (Part A): data conversion and checks

### Conversion (`src/build_pkl.py`)
- The three BuildPkl notebooks are identical except the file name and the key prefix (b1c, b2c, b3c), so one script with a batch argument replaces them.
- Fix needed: h5py 3 removed `Dataset.value`, so `ds[()]` is used instead.
- Memory: one batch at a time, h5py reads lazily, summary mode never touches the within-cycle data. Full mode stops if RAM use goes above 11 GB.
- Summary pickles written first: `data/batch1_summary.pkl` (2.5 MB), `batch2_summary.pkl` (1.6 MB), `batch3_summary.pkl` (3.3 MB). Each holds summary fields, cycle_life and charge_policy per cell.
- Full pickles: see the status line at the end of this section.

### Cell counts and cleaning (`src/load_clean.py`)
- Raw cell counts in the .mat files: batch1 = 46, batch2 = 48, batch3 = 46.
- The handoff numbers (41, 43, 40) are counts AFTER cleaning. After the rules below we get exactly 41, 43, 40 = 124 cells.
- Rules checked line by line against `Load Data.ipynb`:
  - delete b1c8, b1c10, b1c12, b1c13, b1c22 (do not reach 80%).
  - append b2c7, b2c8, b2c9, b2c15, b2c16 to b1c0..b1c4. Cycle numbers of the batch2 part are shifted by the length of the batch1 part. cycle_life increased by 662, 981, 1060, 208, 482. Then delete the batch2 keys. (The notebook comment says "four cells" but the code moves five. We follow the code.)
  - delete b3c37, b3c2, b3c23, b3c32, b3c42, b3c43 (noisy).
- Split exactly as in the notebook (indices into the merged dict): test = 0, 2, ..., 82 plus 83; train = 1, 3, ..., 81; secondary test = batch3. Result: train 41 cells, test 43 cells, secondary test 40 cells. Index 83 is b2c47. This matches the paper (41 / 43 / 40).

### Data quality findings (`src/explore_quality.py`, `src/explore_anomalies.py`)
- IR unit: ohm. Median 0.0166 ohm (16.6 mOhm), 1st to 99th percentile 14.3 to 20.9 mOhm. Plausible for an A123 18650 cell.
- IR missing values: no NaN anywhere. 62 rows with IR exactly 0, all in batch1:
  - cycle 1 of every batch1 cell (41 rows). On this cycle QD, Tavg, Tmin, Tmax and chargetime are also 0. It is a logging artifact.
  - 21 more isolated zeros, mostly near cycle 12 and near cycles 850 to 910 and 1490 to 1520.
- IR stuck: b1c18 has exactly the same IR (0.0199 ohm) for cycles 2 to 53, then drops to about 0.016. A held value, not a measurement.
- Tester pauses: 14 rows with charge time > 60 min (280 to 1290 min), e.g. around cycle 248 in batch2 and cycle 13 in some batch1 cells. Right after a pause, capacity dips and IR jumps for one cycle.
- Capacity spikes: QD up to 2.88 Ah (b1c18 cycle 40), 1.54 (b1c0 cycle 12), 1.49 (b2c12), 1.54 (b2c44); also dips of about 0.1 Ah right after pauses.
- Batch2 cells have about 25 to 35 cycles of data past their listed cycle_life. Kept (they are real cycles past 80%).
- IR level differs by batch: batch3 sits around 15.5 mOhm, batch2 around 17.5 mOhm, batch1 around 16.5 to 17. This is a between-batch offset (likely test setup), so we use R/R0 per cell for the resistance analysis.

### Per-cycle table (`src/preprocess_severson.py`)
Fixed version of the handoff section 11 script. Changes: works on the cleaned merged data, robust references, row cleaning, policy features, split label, Windows-safe paths.

Row cleaning (a row is dropped if any rule fires):

| rule | definition | rows |
|---|---|---|
| zero_row | QD = 0 or Tavg = 0 | 41 |
| ir_zero | IR = 0 (missing) | 62 |
| pause | chargetime > 60 min | 14 |
| qd_spike | abs(QD - rolling median of 11 cycles) > 0.05 Ah | 15 |
| ir_spike | abs(IR - rolling median of 11 cycles) > 0.0015 ohm | 16 |
| ir_stuck | same IR value 5+ cycles in a row | 52 (b1c18 only) |

- Rows: 100,501 total, 149 dropped (overlaps counted once), 100,352 kept. Saved as `outputs/cycle_table.csv` (and `outputs/cycle_table_with_flags.csv` with all rows and flags).
- Decision on IR zeros: dropped, not interpolated. They are only 62 of 100,501 rows, and most sit on cycles that are also bad in other fields.
- Definitions:
  - Q0 = median QD of the first 10 valid cycles from cycle 2 on. R0 = same for IR. Q0 range 1.047 to 1.098 Ah, R0 range 13.2 to 20.1 mOhm.
  - SoH_Q = QD / Q0. Also stored: SoH_Q_nominal = QD / 1.1 Ah.
  - R_ratio = IR / R0. SoH_R_proxy = R0 / IR.
  - Charging policy parsed from the string "C1(Q1%)-C2": C1, Q1, C2, time 0 to 80% SOC, average C-rate 0 to 80%.
- After cleaning: SoH_Q 0.761 to 1.006, R/R0 0.924 to 1.395. At the last cycle of each cell: SoH_Q median 0.814 (0.761 to 0.839), R/R0 median 1.219 (1.081 to 1.329).
- Important for later: Severson stops cells at 0.88 Ah (80% of the 1.1 Ah nominal). Because Q0 is about 1.07 Ah, end of life is SoH_Q about 0.82, not 0.80. So only 748 rows (all batch2) have SoH_Q < 0.8.

### Plots
- `figures/data_capacity_IR_selected_cells.png`: capacity and IR vs cycle for 7 cells across the 3 batches.
- `figures/data_all_cells_overview.png`: all 124 cells, QD, IR, SoH_Q, R/R0 vs cycle.
- `figures/data_cleaning_example.png`: which rows the cleaning removed for b1c2 and b2c44.
- Checkpoint check: counts are right, capacity falls in every cell, IR is never zero after cleaning, no capacity rise beyond the small known early rise (SoH_Q max 1.006). No reason to stop.
- What the plots show: IR is flat or even falls slightly for most of life and then rises sharply near the knee, at the same time as the capacity knee. Batch1 IR is noisy, with steps at the tester pauses.

### Full pickles
- `data/batch1.pkl` (2.9 GB), `batch2.pkl` (2.1 GB), `batch3.pkl` (3.1 GB) written with the within-cycle data, same structure as the notebooks. Peak RAM 3.8 GB, so the 11 GB guard never fired. These hold the raw, uncleaned cells (as in the notebooks). Nothing in Task 2 uses them.

## Task 2 (Part B): ML models and linearity test

Scripts: `src/models.py` (training and evaluation), `src/linearity.py`, `src/plot_models.py`, `src/summarize_results.py`. Full run log in `outputs/models_run.log`.

### Setup
- Data: the clean per-cycle table, rows with cycle >= 21 only (97,947 rows). Reason: the early-cycle features use cycles 2 to 20, so every row only uses information from its own past.
- Split by cell, the paper's split: train 41 cells (27,343 rows), test 43 cells (30,169 rows), batch3 secondary test 40 cells (40,435 rows). Tuning only on train cells with GroupKFold (5 folds, groups = cells). The GroupKFold scores are also reported as a check.
- Algorithms: linear regression (scaled), decision tree, random forest (200 trees), gradient boosting (HistGradientBoosting). Small grids, chosen parameters in `outputs/best_params.csv`.
- Model 1 (SoH_Q). No discharge capacity and no IR in the features, not even early-cycle ones.
  - M1_full (the spec): cycle, current-cycle T_avg, T_max, T_min, charge time; policy (C1, Q1, C2, time 0 to 80%, average C-rate); early-cycle means (cycles 2 to 20) of T_avg, T_max, T_min, charge time.
  - M1_nomeas: same without the current-cycle T and charge time. Added because current-cycle charge time is almost a capacity measurement: less charge goes in as capacity fades. On train cells corr(SoH_Q, charge time) = -0.72, while corr(SoH_Q, T_avg) = -0.10.
- Model 2 (resistance). Features SoH_Q (measured), cycle, T_avg, T_max, T_min. Target R/R0 (main). Also IR in ohm, both directly and as R0 x predicted R/R0. M2_soh uses SoH_Q only (pure curve, for the linearity test).
- Baselines: training mean; per-cycle mean of the training cells; for Model 2 the linear formula R/R0 = 1 + alpha(1 - SoH_Q) with alpha fitted on train cells by least squares (line forced through R/R0 = 1 at SoH = 1).

### Main results (held-out cells; full table in `outputs/results_summary.csv`)

Model 1, target SoH_Q:

| model | test R2 | test RMSE | batch3 R2 | batch3 RMSE | GroupKFold R2 (train) |
|---|---|---|---|---|---|
| baseline, training mean | 0.00 | 0.052 | -0.03 | 0.041 | |
| baseline, per-cycle mean | 0.40 | 0.040 | 0.05 | 0.040 | |
| M1_full linear | 0.81 | 0.022 | -0.07 | 0.042 | 0.73 +/- 0.17 |
| M1_full tree | 0.79 | 0.024 | 0.64 | 0.024 | 0.52 +/- 0.50 |
| M1_full forest | 0.92 | 0.015 | 0.47 | 0.030 | 0.62 +/- 0.52 |
| M1_full gboost | 0.92 | 0.014 | 0.34 | 0.033 | 0.65 +/- 0.43 |
| M1_nomeas forest | 0.75 | 0.026 | -1.98 | 0.071 | 0.60 +/- 0.10 |

Model 2, target R/R0:

| model | test R2 | test RMSE | batch3 R2 | batch3 RMSE | GroupKFold R2 (train) |
|---|---|---|---|---|---|
| baseline, training mean | 0.00 | 0.049 | -0.72 | 0.037 | |
| baseline, per-cycle mean | 0.29 | 0.041 | -2.05 | 0.049 | |
| linear formula, alpha = 0.84 | 0.74 | 0.025 | -0.23 | 0.031 | |
| M2 linear regression | 0.78 | 0.023 | 0.53 | 0.019 | 0.81 +/- 0.07 |
| M2 tree | 0.83 | 0.020 | 0.21 | 0.025 | 0.84 +/- 0.05 |
| M2 forest | 0.85 | 0.019 | 0.52 | 0.020 | 0.86 +/- 0.04 |
| M2 gboost | 0.86 | 0.018 | 0.60 | 0.018 | 0.89 +/- 0.02 |

Resistance in ohm (test cells): R0 x linear formula R2 0.84, RMSE 0.42 mOhm; R0 x M2 forest R2 0.91, RMSE 0.30 mOhm; batch3 0.46 vs 0.79. Predicting IR in ohm directly without R0 is much worse (forest test R2 0.68, batch3 -3.0), because most of the IR spread is the cell-to-cell and batch-to-batch offset in R0, which SoH and temperature cannot explain.

Error by SoH range, MAE on test cells (`outputs/results_by_soh_range.csv`). Rows: SoH >= 0.9: 25,548; 0.8 to 0.9: 4,243; below 0.8: 378 (all 22 batch2 test cells reach it, but only in their last few cycles, past the listed cycle_life; no batch1 or batch3 cell goes below 0.8):

| model | SoH >= 0.9 | 0.8 to 0.9 | below 0.8 | last 10% of life |
|---|---|---|---|---|
| M1_full forest (SoH) | 0.008 | 0.015 | 0.015 | 0.013 |
| M1_nomeas forest (SoH) | 0.012 | 0.034 | 0.073 | 0.038 |
| per-cycle baseline (SoH) | 0.020 | 0.061 | 0.141 | 0.075 |
| M2 forest (R/R0) | 0.011 | 0.021 | 0.026 | 0.022 |
| linear formula (R/R0) | 0.015 | 0.034 | 0.042 | 0.034 |
| per-cycle baseline (R/R0) | 0.022 | 0.052 | 0.144 | 0.068 |

Near end of life (last 10% of each cell's life) the resistance error of the linear formula is 0.034 in R/R0 on test cells and 0.057 on batch3, against 0.022 and 0.027 for the M2 forest. Since end-of-life R/R0 is about 1.2, the formula misses roughly 15 to 30% of the resistance rise there.

Chained pipeline (Model 1 predicted SoH fed into Model 2, test cells): M2 forest R2 0.83 (vs 0.85 with measured SoH), linear formula 0.72 (vs 0.74). On batch3 the chain fails (forest R2 0.03, formula -1.84), because Model 1 already fails there.

### Linearity test (`outputs/linearity_results.csv`, `outputs/alpha_per_cell.csv`)
- Fitted on train cells: alpha = 0.838 (formula through 1). Free-intercept line: R/R0 = 0.994 + 0.892 x. Quadratic: R/R0 = 1.002 + 0.394 x + 3.03 x^2, where x = 1 - SoH_Q.
- R2 (train / test / batch3): formula 0.785 / 0.738 / -0.227; free line 0.793 / 0.746 / -0.115; quadratic 0.823 / 0.781 / 0.129; forest on SoH only 0.835 / 0.792 / 0.157; forest on SoH + cycle + T 0.952 / 0.850 / 0.516.
- Residual pattern (`figures/linearity_residuals.png`, binned over all cells): the formula over-predicts R/R0 by 0.01 to 0.03 for SoH between 0.84 and 1.0, then under-predicts by about 0.04 below SoH 0.82. That is a systematic bend, not noise: R stays almost flat for most of life and climbs fast after the knee. The forest on SoH only removes most of this bend (residuals within -0.02 to 0).
- Alpha per cell is far from constant: mean 0.70, std 0.29 (CV 0.42). By batch, median 0.80 (batch1), 0.92 (batch2), 0.38 (batch3). For 16 of 124 cells even a cell-specific straight line gives R2 < 0.5.
- **Verdict:** the linear form is a reasonable first-order approximation inside one batch of cells (R2 about 0.74 on test cells, close to the free line), and it gets the direction and rough size right (R/R0 about 1.15 to 1.2 at SoH 0.8). But it is not a good description of the shape: the true curve is convex (flat, then a knee), so the linear form is too pessimistic in mid-life and too optimistic near end of life, which is exactly where the power limit matters. Alpha is also not transferable between test campaigns (batch3 alpha is about half). A quadratic in (1 - SoH) is a simple and better alternative, and adding cycle and temperature (Model 2) helps more.

### Feature importance (`figures/feature_importance.png`, permutation importance on test cells)
- M1_full forest: charge time (0.032 RMSE increase) and cycle (0.030) dominate; then early charge time and policy features (about 0.005 each); temperatures almost nothing.
- M1_nomeas forest: cycle (0.053) dominates, then C2, C1 and other policy features (0.003 to 0.006).
- M2 forest: SoH_Q (0.050), then cycle (0.013), T_min (0.0025); T_avg and T_max are near zero.

### Figures
- `m1_full_pred_vs_actual.png`, `m1_nomeas_pred_vs_actual.png`, `m2_pred_vs_actual.png`, `m2_formula_pred_vs_actual.png`: predicted vs actual, train (blue) and test / batch3 (red) in separate labelled panels with R2 and RMSE.
- `error_vs_soh.png`: MAE vs SoH bin on held-out cells, both models, with baselines.
- `linearity_R_vs_SoH.png`: R/R0 vs SoH with linear formula, quadratic and forest; `linearity_residuals.png`; `linearity_alpha_per_cell.png`.
- `feature_importance.png`, `example_trajectories.png` (SoH and R/R0 vs cycle for 5 cells).

### Saved outputs
- Models: `outputs/models/*.joblib` (19 files, 23 MB).
- Tables: `outputs/results_metrics.csv` (every model x split x subset), `results_summary.csv`, `results_by_soh_range.csv`, `cv_results.csv`, `best_params.csv`, `linearity_results.csv`, `alpha_per_cell.csv`, `feature_importance.csv`, `predictions.csv`, `alpha.json`.

## Summary for the report (Task 1 + 2)

What worked
- Conversion and cleaning reproduce the paper exactly: 124 cells, split 41 / 43 / 40.
- Model 1 (SoH from operating data, no capacity or IR inputs) reaches R2 0.92 and RMSE 0.015 on the 43 test cells with random forest or gradient boosting. Both beat the trivial baselines by a wide margin (per-cycle mean R2 0.40).
- Model 2 (R/R0 from SoH, cycle and temperature) reaches R2 0.85 to 0.86 on test cells, better than the linear formula (0.74), and the gap is largest near end of life.
- The linearity test gives a clear answer: alpha = 0.84 on train cells, but the real R vs SoH curve is convex, and alpha varies strongly between cells and batches.

What did not work, or only partly
- Generalisation to batch3 is weak for every model (Model 1 R2 -0.07 to 0.64, Model 2 0.21 to 0.60). Batch3 was run a year later with a different test structure ("newstructure" in the policy name). Its cells live longer for similar policies, and their IR is lower and rises less (alpha about 0.38). Train cells come only from batches 1 and 2.
- Model 1 leans heavily on the current-cycle charge time. That works as an online estimator (a BMS also measures charge time), but it is not a forecast. Without the current-cycle measurements (M1_nomeas), test R2 falls to 0.75 and batch3 fails (R2 -2.0). So charging policy plus cycle number alone cannot predict SoH well for unseen cells.
- GroupKFold scores on train cells vary a lot between folds for Model 1 (R2 0.62 +/- 0.52). With only 41 cells, one or two unusual cells in a fold (for example the very long-lived b1 cells) change the score a lot. Model 2 is much more stable (0.86 +/- 0.04).
- Tree models overfit Model 1 on train rows (forest train R2 0.999 vs CV 0.62). The test score is still good, but the CV spread is the more honest uncertainty.

Limitations
- LFP only (A123 APR18650M1A, 1.1 Ah). No NMC or NCA. Nothing here says anything about other chemistries.
- Cell level only. Scaling to a battery system is not done here.
- The main thing varied in this dataset is the fast-charging policy. According to the Severson paper (not checked in our data), ambient temperature was fixed (chamber at 30 C) and all cells had the same discharge (4C to 2.0 V). Our data agrees with this: measured cell temperatures sit in a narrow range (Tavg 1st to 99th percentile 29.7 to 38.2 C). So temperature here is measured cell temperature, which mostly follows the charging policy, and depth of discharge does not vary at all. We cannot learn temperature or DoD effects from Severson.
- IR in the summary is the value reported by the tester once per cycle. We did not check how it was measured (pulse length, SOC), so it may not equal the DC resistance that sets peak power over seconds.
- End of life in this dataset is QD = 0.88 Ah, which is SoH_Q about 0.82 with our Q0. So the "below 0.8" range has only 378 test rows, all from the last few cycles of batch2 cells. Results there are thin.
- Rows from long-lived cells outnumber rows from short-lived cells. Metrics are pooled over rows, so long-lived cells weigh more.

Assumptions (all of them)
1. Cleaning and split follow `Load Data.ipynb` exactly (five merged cells, not four).
2. Row cleaning rules and thresholds as in the Task 1 table (0.05 Ah, 0.0015 ohm, 60 min, 5-cycle stuck run). Thresholds were picked by looking at the data, not tuned on model results.
3. IR = 0 means missing. Those rows are dropped, not filled.
4. Q0 and R0 = median of the first 10 valid cycles from cycle 2 on.
5. SoH_Q = QD / Q0 (not QD / 1.1 Ah). SoH_Q can be slightly above 1 early in life (max 1.006).
6. Modelling rows start at cycle 21, so early-cycle features (cycles 2 to 20) are always in the past.
7. Early-cycle features for b1c18 use cycles 2 to 20 although its IR is stuck there, because only T and charge time are used.
8. Policy parsed from the string; "4C(31%)-5" (missing "C") read as C2 = 5. The 0 to 80% charge time assumes the two-step CC policy as written, without CV.
9. Model 2 is trained and tested on measured SoH_Q. The chained result shows what happens with predicted SoH.
10. The linear formula is fitted as a single pooled alpha over training rows, forced through R/R0 = 1 at SoH = 1, as in Chen, Liu & Fu.
11. Random seeds fixed at 0. The models were not tuned on the test cells.

Still open after Task 2
- Sandia data for a chemistry comparison, and solar/wind data, are still not obtained.

## Task 3: power cap and swing equation simulation

Scripts: `src/power_sim.py` (curves, power and energy limits, simulation, figures), `src/task3_summary.py` (tables). Run log `outputs/task3_run.log`, compact tables `outputs/task3_summary.txt`. All parameters saved in `outputs/task3_params.json`.

Small change to Task 2 code: `src/linearity.py` now also saves the fitted coefficients to `outputs/resistance_curve_coefs.json` (`linearity_results.csv` only had the scores). Rerun gave the same values as before.

### 1. Resistance curve R(SoH)
x = 1 - SoH_Q. Every curve is divided by its own value at SoH = 1, so all start at exactly 1 (the quadratic's raw value at SoH = 1 is 1.0018).

| curve | source | R/R(1) at SoH 0.9 | 0.82 | 0.70 (extrapolated) |
|---|---|---|---|---|
| quadratic (MAIN): (1.0018 + 0.3941x + 3.0280x^2)/1.0018 | our training cells | 1.070 | 1.169 | 1.390 |
| linear formula: 1 + 0.838x | our training cells | 1.084 | 1.151 | 1.252 |
| random forest, SoH only (check) | our training cells | 1.066 | 1.180 | not defined |
| literature +30% at EOL: 1 + 1.5x | NOT our data | 1.150 | 1.270 | 1.450 |
| literature +100% at EOL: 1 + 5.0x | NOT our data | 1.500 | 1.900 | 2.500 |

- The literature curves are straight lines that reach +30% or +100% of R0 at SoH = 0.8 (the usual end-of-life point). They are a sensitivity case only.
- Check against the forest curve over SoH 0.76 to 1.0: quadratic RMSE 0.011 (max diff 0.031), linear RMSE 0.026 (max 0.052), literature +30% RMSE 0.082 (max 0.122), literature +100% RMSE 0.567 (max 0.962). So the quadratic matches the data-driven curve best, the linear formula is close, and the +100% literature case is far from what our cells do.
- Extrapolation: our data covers SoH_Q 0.761 to 1.006. Anything below 0.76 is flagged `extrapolated` in the code and the CSVs, shaded grey in the figures, and dotted in the time plots. The forest curve is set to NaN below 0.76 (a forest cannot extrapolate).

### 2. Power and energy limits
- P_max = V_min (V_oc - V_min) / R, with V_oc = 3.30 V and V_min = 2.00 V (assumed LFP values). So P_max/P_max_fresh = R(1)/R(SoH) = 1/r.
- P_max fraction (quadratic / linear / lit +30% / lit +100%): SoH 0.9: 0.935 / 0.923 / 0.870 / 0.667; SoH 0.82: 0.856 / 0.869 / 0.787 / 0.526; SoH 0.7 (extrapolated): 0.719 / 0.799 / 0.690 / 0.400.
- For reference only, one cell with the median R0 of our 124 cells (16.60 mOhm, range 13.20 to 20.12): fresh P_max = 156.7 W at 78.3 A. Quadratic: 146.5 W at SoH 0.9, 134.0 W at 0.82, 112.7 W at 0.7. This is a theoretical resistance limit at a very high current. Real cells and inverters have lower current limits, which we do not model. That is why the simulation only uses the fraction.
- Energy limit: the battery keeps a reserve for inertia support equal to T_res seconds of its fresh rated power, times SoH_Q (capacity fade). Main case T_res = 10 s, tight case T_res = 1 s. When the reserve is used up the battery cannot discharge any more (it can still charge).
- `figures/task3_R_vs_SoH.png` and `figures/task3_Pmax_vs_SoH.png` (the second also shows the energy fraction = SoH_Q).

### 3. Simulation setup (all per unit on the system load base, f0 = 50 Hz)
- Swing equation: 2H dw/dt = Pm - dP_loss - D w + P_bat, solved with explicit Euler, dt = 1 ms, 60 s.
- Low-inertia grid: renewable (inverter) share 60%, synchronous share 40% with H = 5 s on its own base, so system H = 2.0 s. Load damping D = 1.
- Governor of the remaining synchronous plants: 5% droop on their 40% capacity (gain 8 pu/pu on system base), one lumped first-order lag T = 3 s. No secondary control (AGC), so frequency settles at -dP/(K + D): -0.28 Hz (5% loss) and -0.56 Hz (10% loss).
- Battery (grid-forming inverter): virtual inertia P_req = -2 Hv dw/dt with Hv = 2.0 s (fixed). P_bat = P_req clipped to +/- P_max(SoH) and to the energy left. H, Hv and the DC-link capacitor are not changed with SoH. Only the cap and the energy limit change. The capacitor itself is not modelled (it only matters for the first milliseconds, and it does not age with the battery).
- Battery size: the fresh P_max is set from the 10% event so that the fresh peak request (Hv/(H+Hv) x 0.10 = 0.05 pu, at t = 0) divided by fresh P_max equals the demand ratio d = 0.5, 0.8, 1.0, 1.2. The 5% event uses the same battery, so its demand ratio is half (0.25, 0.4, 0.5, 0.6). If each event had its own battery, the results would be the same in relative terms, because the model is linear apart from the cap.
- Cases: 2 events x 4 demand ratios x 2 energy reserves x 5 curves x SoH {1.0, 0.9, 0.82, 0.7 extrapolated} (forest only to 0.82). Sweeps for figures: SoH 0.60 to 1.00 in steps of 0.01, and demand ratio 0.3 to 1.6.
- Metrics: frequency nadir; RoCoF at t = 0 and the largest RoCoF over a 500 ms window; settling time (last time outside 5% of the final deviation); cap saturation (yes/no and how long); energy reserve run out (yes/no).
- "Aging makes no difference" = nadir and 500 ms RoCoF both within 0.001 of the fresh battery with the same size.

### 4. Results
References (no SoH effect):

| | nadir | RoCoF t = 0 | RoCoF 500 ms | settling |
|---|---|---|---|---|
| 10% loss, no battery | 48.683 Hz | 1.25 Hz/s | 1.144 Hz/s | 11.4 s |
| 10% loss, uncapped battery | 48.970 Hz | 0.625 Hz/s | 0.598 Hz/s | 15.5 s |
| 5% loss, no battery | 49.341 Hz | 0.625 Hz/s | 0.572 Hz/s | 11.4 s |
| 5% loss, uncapped battery | 49.485 Hz | 0.313 Hz/s | 0.299 Hz/s | 15.5 s |

A fresh battery with d <= 1.0 behaves exactly like the uncapped one. The battery halves RoCoF and raises the nadir by 0.29 Hz (10% loss). It also makes settling slower (15.5 vs 11.4 s), because pure virtual inertia also pushes back against the recovery (the battery charges while frequency rises).

10% loss, energy reserve 10 s: nadir (Hz) / RoCoF 500 ms (Hz/s); S = cap saturates:

| curve, SoH | d = 0.5 | d = 0.8 | d = 1.0 | d = 1.2 |
|---|---|---|---|---|
| fresh (any curve) | 48.970 / 0.598 | 48.970 / 0.598 | 48.970 / 0.598 | 48.967 / 0.667 S |
| quadratic 0.9 | 48.970 / 0.598 | 48.970 / 0.598 | 48.970 / 0.612 S | 48.963 / 0.698 S |
| quadratic 0.82 | 48.970 / 0.598 | 48.970 / 0.598 | 48.968 / 0.654 S | 48.957 / 0.736 S |
| quadratic 0.7 (extrap.) | 48.970 / 0.598 | 48.969 / 0.629 S | 48.957 / 0.732 S | 48.939 / 0.801 S |
| linear 0.82 | 48.970 / 0.598 | 48.970 / 0.598 | 48.968 / 0.647 S | 48.958 / 0.730 S |
| lit +30% 0.82 | 48.970 / 0.598 | 48.970 / 0.599 S | 48.964 / 0.693 S | 48.949 / 0.768 S |
| lit +100% 0.82 | 48.970 / 0.598 | 48.949 / 0.767 S | 48.923 / 0.843 S | 48.899 / 0.893 S |
| lit +100% 0.7 (extrap.) | 48.965 / 0.686 S | 48.916 / 0.858 S | 48.886 / 0.915 S | 48.861 / 0.953 S |

Main findings
1. **Aging only matters when the demand is close to the cap.** Saturation starts when d x R(SoH)/R(1) > 1. With our data curves at SoH 0.82 (r about 1.15 to 1.18), aging makes no difference at all for d = 0.5 and 0.8. It only shows up at d >= 1.0 (roughly d above 0.86). Counted over all aged cases (10% loss, 10 s reserve): no difference in 13 of 14 at d = 0.5, 8 of 14 at d = 0.8, 0 of 14 at d = 1.0 and 1.2.
2. **For the 5% loss with the same battery, aging makes no difference** for any data-backed curve, even at SoH 0.7 (extrapolated). Only literature +100% at SoH 0.82 and below shows a small effect (nadir -0.001 Hz, RoCoF 0.299 to 0.321 Hz/s at d = 0.6).
3. **The cap hits RoCoF much more than the nadir.** Virtual inertia power is largest at t = 0 and falls steadily, reaching zero exactly at the nadir (about 3 s, where dw/dt = 0). So the cap only clips the first 0.5 to 1.5 s (see `task3_frequency_response.png`), while the nadir is set mainly by the governor. At SoH 0.82 and d = 1.0 with the quadratic curve, RoCoF rises from 0.598 to 0.654 Hz/s (+9%) but the nadir only drops by 0.002 Hz. At d = 1.2 the change vs a fresh battery of the same size is RoCoF 0.667 to 0.736 Hz/s (+10%) and nadir -0.010 Hz.
4. **The choice of R(SoH) curve matters more than the quadratic vs linear question.** Within the data range the quadratic and linear curves give almost the same result (RoCoF 0.654 vs 0.647 at SoH 0.82, d = 1.0). They only separate in the extrapolated range (SoH 0.7: 0.732 vs 0.687). The literature +100% case gives a much larger effect (0.843 Hz/s and nadir -0.047 Hz vs fresh at SoH 0.82, d = 1.0), but our data does not support that much resistance rise for these LFP cells.
5. **Energy (capacity fade) matters only if the reserve is tight.** With a 10 s reserve the battery used at most 20% of what was available (quadratic, 10% loss), so the energy limit never bound. With a 1 s reserve the energy runs out in almost every case with d >= 0.8, even for a fresh battery, and then the nadir drops a lot (10% loss, d = 0.8: fresh 48.820 Hz, quadratic SoH 0.82 48.769 Hz, compared with 48.970 Hz with 10 s). For the 5% loss with a 1 s reserve and d = 0.6, SoH 0.82 gives 49.426 Hz vs 49.485 Hz fresh. Here capacity fade matters more than resistance rise.
6. Odd but real side effect: with a 1 s reserve and the literature +100% curve, an aged battery can give a slightly better nadir than a fresh one (up to +0.06 Hz), because its lower cap spreads the small energy reserve over a longer time. This shows the result depends on how the reserve is managed, not only on the cap.
7. Settling time barely changes with SoH (15.5 s fresh, 15.2 to 15.5 s aged, reserve 10 s). Not a useful metric for aging here.

Figures
- `figures/task3_R_vs_SoH.png`: five R(SoH) curves plus binned data, extrapolation shaded.
- `figures/task3_Pmax_vs_SoH.png`: P_max fraction for the five curves plus the energy fraction.
- `figures/task3_frequency_response.png`: frequency and battery power vs time, 10% loss, d = 1.0, SoH 1.0 / 0.9 / 0.82 / 0.7 (dotted, extrapolated), quadratic curve vs literature +100%.
- `figures/task3_nadir_rocof_vs_SoH_10pct.png` and `_5pct.png`: nadir and RoCoF vs SoH for each demand ratio and curve.
- `figures/task3_nadir_rocof_vs_demand.png`: nadir and RoCoF vs demand ratio for SoH 1.0 / 0.9 / 0.82 / 0.7.
- Tables: `outputs/task3_cases.csv` (all 304 cases), `task3_references.csv`, `task3_sweep_soh.csv`, `task3_sweep_demand.csv`, `task3_curves.csv`.

### Assumptions (Task 3)
1. V_oc = 3.30 V and V_min = 2.00 V, constant with SoH and SOC. Not measured from our data.
2. P_max fraction = R(1)/R(SoH). The P_max formula is the standard Thevenin/BMS form. Cell to system scaling is avoided by using the fraction only.
3. R(SoH) curves normalised to 1 at SoH = 1. The main curve is the quadratic fitted on training cells. The literature curves are linear, reach +30% or +100% at SoH 0.8, and are not from our data.
4. Our data covers SoH 0.76 to 1.0. SoH 0.7 and all curve values below 0.76 are extrapolation.
5. Energy reserve = T_res x fresh rated power x SoH_Q, with T_res = 10 s (main) or 1 s (tight). These T_res values are our choice, not from data.
6. Grid: 60% renewable share, H_sync = 5 s, system H = 2.0 s, D = 1, 5% droop on synchronous capacity, single 3 s governor lag, no AGC. These are typical textbook-style values, not taken from a real grid. No real solar/wind data was used. The disturbance is a step loss of 5% or 10% of generation at t = 0.
7. Battery control: pure virtual inertia (P = -2 Hv dw/dt, Hv = 2 s), no extra fast frequency response or droop, no measurement filter or PLL delay, instant power response. Symmetric cap for discharge and charge.
8. The battery is sized against the 10% event via the demand ratio, and the same battery is used for the 5% event.
9. Explicit Euler with dt = 1 ms. RoCoF over a 500 ms window is the main RoCoF number; the t = 0 value is also in the CSV.

### Limitations (Task 3)
- A simple single-area model with one lumped governor. No network, no voltage dynamics, no inverter current limit, no converter or capacitor dynamics.
- The real binding limit in practice is often the inverter current rating, not the battery resistance. If the inverter is sized below the battery's P_max, aging does not matter until P_max(SoH) falls below the inverter rating.
- R from the Severson summary IR (measurement method not checked, see Task 2), for LFP cells only, cell level.
- Pure virtual inertia does little for the nadir by design. A battery that also gives fast frequency response (droop) would draw power for longer, so the cap and the energy limit would affect the nadir more. Not simulated.
- Aging is applied as a fixed SoH for each run. There is no event-by-event degradation (stretch goal in HANDOFF).

## Solar and wind data (24 Oct 2026): downloaded and checked

Downloaded by the user from Renewables.ninja (free account), checked with `src/check_renewables.py`. Figure: `figures/renewables_data_check.png`.

- Files: `data/ninja_pv_23.2500_69.6703_uncorrected.csv`, `data/ninja_wind_23.2500_69.6703_uncorrected.csv`.
- Location: 23.25 N, 69.67 E (Kutch, Gujarat). Year 2025. Dataset MERRA-2 reanalysis, Renewables.ninja version 1.4. Times in UTC.
- Solar settings: 1 kW, system loss 0.1, fixed (no tracking), tilt 23, azimuth 180. Includes raw irradiance and temperature.
- Wind settings: 1 kW, hub height 100 m, turbine Vestas V90 2000. Includes wind speed.
- License CC BY-NC 4.0. Cite the references in the file headers: doi 10.1016/j.energy.2016.08.060 (solar) and 10.1016/j.energy.2016.08.068 (wind).

Checks (all passed):
- 8,760 hourly rows each (2025 is not a leap year), 1 Jan 00:00 to 31 Dec 23:00 UTC. No gaps, no duplicate times, no missing values, no negative values. Both files have exactly the same timestamps.
- Time zone: mean solar output peaks at 07:00 UTC = 12:30 IST, and solar is exactly 0 between 20:30 and 04:30 IST. So the UTC label is right, and we shift by +5:30 to get local time.
- Solar: annual capacity factor 0.198, maximum 0.815. Monthly capacity factor lowest in the monsoon months (0.169 to 0.173 in June to August) and highest in March and April (0.23).
- Wind: annual capacity factor 0.328, maximum 0.980. Strong April to September (0.34 to 0.51 by month), weak in November and December (0.11 and 0.16). On average windier in the evening (0.40 at 19 h IST) than in the morning (0.26 at 9 h).
- Wind power follows a smooth power curve vs wind speed (near 0 below 3 m/s, about 0.58 at 8 to 10 m/s, close to 1 above 12 m/s).
- 451 hours of the year have both solar and wind below 5% of capacity.

Caveats:
- "uncorrected": this wind series has no bias correction against measured output. MERRA-2 wind speeds can be biased. We did not compare against real plant output for this site, so the wind capacity factor should be read as a modelled value.
- These are capacity factors of 1 kW of modelled plant at one point. They are not real grid generation or demand. Turning them into a renewable share for each hour needs an assumed grid mix (next step).

## Real Indian grid data (24 Oct 2026): downloaded and checked

Source: "Electricity Demand, Solar and Wind Generation Data (September 2021 - June 2025) of India at 1-hour interval", Mendeley Data v2, DOI 10.17632/y58jknpgs8.2, license CC BY 4.0. Data provided by NERLDC, Grid-India (Ministry of Power). Contributors D. Mukherjee, K. Kalita, S. Kumar (IIT Guwahati, Power Grid Corporation). Downloaded by the user into `data/demand/` (30 Excel files).

What the files actually contain (more than the page says):
- Sep 2021 to Sep 2022: 5-minute all-India SCADA values. Oct 2022 to Dec 2023: 10-second values. Both have demand, thermal (coal), hydro, gas, nuclear, wind, solar (MW), plus three tags whose names say inertia (INTRA_SYSTEM_INTERTIA, ER_Total_Inertia, SYS_IN).
- Jan 2024 to 24 Jun 2025: hourly sheet with demand, wind, solar, total generation only (no split by fuel, no inertia tags). The file ends on 24 June 2025, not 30 June. It is not stated whether these hourly values are averages or snapshots.

Conversion (`src/build_grid_hourly.py`): every file reduced to hourly means. 33,432 hours, 1 Sep 2021 00:00 to 24 Jun 2025 23:00, no missing hours, no duplicated hours. Each SCADA month is complete (8,640 to 8,928 five-minute rows, or 241,920 to 267,840 ten-second rows). 6 non-date footer rows dropped from the 2024 to 2025 file (Average, Maximum, Minimum, Sum, header). Log: `outputs/grid_india_files.csv`.

Checks (`src/check_grid_data.py`, `src/check_grid_outliers.py`):
- Timestamps are IST: mean solar peaks at 12:00 file time in all three periods.
- Consistency: (sum of listed sources) / demand has median 1.010 (1st to 99th percentile 0.992 to 1.027). Total generation / demand in 2024 to 2025 also has median 1.010. So the numbers add up.
- No jump at the two format changes (daily demand, wind and solar continue smoothly across 1 Oct 2022 and 1 Jan 2024).
- Problems found: 12 all-zero hours (SCADA outage); nuclear stuck at 18 to 38 GW on 4 and 5 Sep 2021; a wind spike of 38.7 GW on 11 Aug 2022 (sources 13% above demand); hours with part of the data missing (for example demand 80 GW between hours of about 150 GW); frozen values in 2024 to 2025 (for example solar exactly 64,300.61 MW from 00:00 to 04:00 on 7 Mar 2025, wind and total generation repeated for hours).

Cleaning (`src/clean_grid_data.py`), an hour is dropped if any rule fires:

| rule | definition | hours |
|---|---|---|
| zero | demand <= 0 | 12 |
| dropout | demand more than 15% away from its 5-hour rolling median | 23 |
| balance | sources (or total generation) / demand outside 0.95 to 1.08 | 215 |
| stuck | demand, wind, total generation, or daytime solar (> 500 MW) exactly repeated 3+ hours | 337 (330 of them in 2024 to 2025) |
| night_solar | solar > 1000 MW between 21:00 and 04:59 IST | 298 |

- 740 hours dropped (2.21%), 32,692 kept. Saved in `outputs/grid_india_hourly_clean.csv` (all hours with flags).
- Small negative solar values at night (down to -75 MW, a sensor offset) are set to 0.

Clean results:
- Variable renewable share (wind + solar) / demand by year, median / 95th percentile / max: 2021 0.060 / 0.182 / 0.241; 2022 0.077 / 0.217 / 0.316; 2023 0.083 / 0.239 / 0.311; 2024 0.079 / 0.263 / 0.333; 2025 (to June) 0.090 / 0.321 / 0.396. Highest clean hour 0.396 at 12:00 on 2 Jun 2025 (wind 23.9 GW + solar 57.6 GW of 205.4 GW demand).
- The share peaks around midday (solar) and grows each year. Average midday share rises from about 0.17 (2021) to about 0.31 (2025).
- Synchronous share (thermal + hydro + gas + nuclear) / demand, 2021 to 2023: minimum 0.69 to 0.76, median 0.93 to 0.95. For 2024 to 2025 it is computed as 1 - renewable share (no split by fuel); minimum 0.60 (2025). Check: in 2021 to 2023 where both exist, the two ways differ by +0.0097 on average (max 0.074).
- Figures: `figures/grid_india_data_check.png` (raw), `figures/grid_india_clean.png` (clean).

Inertia-labelled tags: non-zero only from mid-December 2021 to December 2023. INTRA_SYSTEM_INTERTIA median 3.33 (0.28 to 3.79), SYS_IN median 2.69 (0.25 to 3.73), ER_Total_Inertia median 3.62 (up to 17.5, and the same value repeats in 17,228 of 17,538 hours, so it is mostly frozen). The files give no unit or definition. SYS_IN has only a weak correlation with synchronous generation (0.18). We do not know what these numbers are, so we do not use them as an input. They could be inertia constants in seconds, but that is a guess and is not used anywhere.

What this means for the simulation (a calculation, not yet simulated): with H = 5 s per synchronous plant and H_system = 5 s x synchronous share, the lowest clean synchronous share (0.60, 2025) gives H of about 3.0 s. The Task 3 battery was sized at H = 2.0 s. So on the real all-India grid the inertia request per unit of lost generation is smaller than in Task 3, and the battery would saturate less often.

## Simulation over the real Indian grid hours (24 Oct 2026)

Script `src/real_grid_sim.py`, log `outputs/real_grid_run.log`, parameters `outputs/real_grid_params.json`.
Small change to `src/power_sim.py`: `simulate()` now accepts per-case inertia and governor gain. With the defaults it gives exactly the Task 3 numbers (checked in the script: no-battery nadir -0.6587 / -1.3174 Hz, same as before).

Why this run: the abstract promises that "real-world solar and wind generation data are used to construct a representative operating scenario". Here the operating condition of every hour comes from Grid-India's measured generation and demand.

Setup
- 32,692 clean hours (Sep 2021 to Jun 2025).
- Synchronous share per hour = (thermal + hydro + gas + nuclear) / demand to 2023, and 1 - (wind + solar) / demand for 2024 to 2025. Capped at 1.0 (218 hours were slightly above 1).
- System inertia H(t) = 5 s x synchronous share (5 s per synchronous plant is an assumption; the data gives MW, not H). Governor gain = synchronous share / 0.05 droop. D = 1, governor lag 3 s, as in Task 3.
- Resulting H, by year (median / 1st percentile / min): 2021 4.74 / 3.98 / 3.82 s; 2022 4.67 / 3.82 / 3.46; 2023 4.64 / 3.70 / 3.52; 2024 4.61 / 3.51 / 3.34; 2025 4.55 / 3.24 / 3.02 s.
- Comparison only (not calibration): Grid-India reports a system inertia of 5 to 9 s (NASPI 2023 slides, full report not yet read). Our hourly H is at or below that range, so our inertia is on the cautious (low) side.
- Battery: virtual inertia Hv = 2 s, R(SoH) curves from Task 3, energy reserve 10 s x SoH.
- Battery size: set against the lowest-inertia clean hour in the data (2 Jun 2025 12:00, synchronous share 0.6035, H = 3.02 s). In that hour the peak inertia request for a 10% loss is 0.0398 pu; fresh P_max = 0.0398 / d, d = 0.8, 1.0, 1.2. The same battery is used in every hour and for the 5% loss. Task 3's size (set for H = 2 s) is not used, because the real grid never gets that low.
- Speed: synchronous share rounded to 0.005 (79 distinct values, max rounding 0.0025). Each value simulated once (5,846 cases) and mapped back to the hours.

Results
- Without a battery: 10% loss nadir 49.06 to 49.39 Hz, RoCoF 0.48 to 0.77 Hz/s; 5% loss nadir 49.53 to 49.70 Hz, RoCoF 0.24 to 0.39 Hz/s (range over all real hours).
- **5% loss: aging never changes the response** for any of our data curves (quadratic, linear), any battery size, any SoH including 0.7. Only the literature +100% curve shows an effect (for example SoH 0.82, d = 1.2: 1.9% of hours, RoCoF up by at most 0.013 Hz/s).
- **10% loss, quadratic curve (main), share of hours where aging changes the response** (nadir or RoCoF differs from a fresh battery by more than 0.001):

| battery size d | SoH | 2021 | 2022 | 2023 | 2024 | 2025 (Jan to Jun) |
|---|---|---|---|---|---|---|
| 0.8 | 0.9 / 0.82 | 0% | 0% | 0% | 0% | 0% |
| 1.0 | 0.9 | 0% | 0% | 0% | 0% | 1.5% |
| 1.0 | 0.82 | 0% | 0.4% | 1.5% | 7.9% | 22.4% |
| 1.2 | 0.9 | 18.5% | 26.4% | 30.7% | 33.4% | 38.0% |
| 1.2 | 0.82 | 67.5% | 73.3% | 77.5% | 85.9% | 93.3% |
| 1.0 | 0.7 (extrapolated) | 57.9% | 65.1% | 69.5% | 76.1% | 81.4% |

- The share grows every year because renewables grow. Within a day it is a midday effect: in 2025 with d = 1.0, the SoH 0.82 battery hits its cap in almost every hour from 11:00 to 13:00 and in no hour at night (`figures/real_grid_2025_by_hour.png`).
- **The size of the effect is small.** Worst hour in the data (2 Jun 2025 11:00, H = 3.02 s), 10% loss, d = 1.0: RoCoF 0.478 Hz/s fresh vs 0.509 Hz/s at SoH 0.82 (+6.6%), nadir 49.2244 vs 49.2228 Hz (-0.002 Hz). Mean RoCoF over 11:00 to 13:00 in 2025: 0.439 fresh, 0.440 (SoH 0.9), 0.448 (SoH 0.82), compared with 0.678 Hz/s without a battery. So even an aged battery keeps most of the benefit.
- Linear vs quadratic curve: similar within the data range (d = 1.0, SoH 0.82: 3.8% vs 5.3% of all hours). Literature +100%: aging matters in 100% of hours at SoH 0.82 for d >= 0.8, with RoCoF up to 0.13 Hz/s higher. Our data does not support that much resistance rise.
- Energy reserve (10 s) was never used up in any case.

Figures
- `figures/real_grid_inertia.png`: hourly H from real data vs Grid-India's reported 5 to 9 s, and the sizing hour.
- `figures/real_grid_aging_hours_by_year.png`: % of hours where aging matters, by year, for the three battery sizes and three R(SoH) curves.
- `figures/real_grid_rocof_vs_share.png`: RoCoF vs non-synchronous share, fresh vs aged.
- `figures/real_grid_2025_by_hour.png`: 2025 average day: RoCoF and % of hours at the cap.
- Tables: `outputs/real_grid_summary.csv` (per year and overall), `outputs/real_grid_hourly.csv` (per hour, main case), `outputs/real_grid_bins.csv` (all simulated cases).

Assumptions added by this step
1. H = 5 s per unit of synchronous generation (not in the data).
2. Online synchronous capacity = synchronous generation (no spare spinning units), which makes H low (cautious).
3. 2024 to 2025: synchronous share = 1 - (wind + solar) / demand (no fuel split in those files).
4. Battery sized for the lowest-inertia hour in the data; same battery in all hours.
5. Each hour is treated as one operating state; a 5% or 10% loss is applied to it as a test. These are not real events in those hours.

Limitations found (comparison against reported values, not fixed)
- Our governor response is much stronger than the real grid's. With 5% droop on the synchronous share, steady-state response is (synchronous share / 0.05 + 1) pu per pu, about 19.6 at a typical share of 0.93. For 180 to 200 GW of demand at 50 Hz that is roughly 70,000 to 80,000 MW per Hz, while Grid-India reports about 10,000 MW per Hz (NASPI 2023 slides). Our time to nadir is about 3 s vs the reported 9 to 14 s. So our nadirs are too optimistic, and the real effect of aging on the nadir is likely larger than shown here. The RoCoF result is the more reliable one, because the first second is set mainly by inertia, not the governor.
- All-India level only. A high-renewable region would have lower inertia than the national average.

## Sandia cycling data (24 Oct 2026): downloaded and SoH built

Why: the abstract promises SoH "as a function of operating factors such as chemistry, temperature, depth of discharge, and cycling stress". Severson varies only the fast-charge policy (LFP only, fixed chamber temperature, full DoD). Sandia varies chemistry, temperature, DoD and discharge rate.

Source and download
- Study: Preger et al. 2020, J. Electrochem. Soc., "Degradation of commercial lithium-ion cells as a function of chemistry and cycling conditions" (Sandia National Laboratories). Data source to cite: www.batteryarchive.org.
- Downloaded with `src/download_sandia.py` from the public Battery Archive "Cycle Test Data" dashboard (the same requests the web page makes, using the dashboard's public link key). Per cycle: ah_c, ah_d, e_c, e_d (query 26), v_max, v_min, v_c_mean, v_d_mean (query 40), ah_eff, e_eff (query 28). Raw responses kept unchanged in `data/sandia/cycle/raw/`, one CSV per cell in `data/sandia/cycle/`.
- No resistance in this data. Within-cycle voltage is only available for a few chosen cycles, every 2 minutes, without current, so a pulse resistance cannot be computed. Full raw data is only by email request (not possible for us).
- 86 cells, all commercial 18650: 30 LFP, 32 NMC, 24 NCA. 33 conditions: temperature 15 / 25 / 35 C; SOC window 0-100, 20-80, 40-60 %; charge 0.5C; discharge 0.5, 1, 2, 3C; 1 to 4 replicate cells per condition. 2,958 to 19,173 cycles per cell (LFP), roughly 1 to 2 years of testing.

How the test runs (seen in the data)
- Cycles 1 to 3: capacity check (full voltage window, 0.5C charge and discharge, about 4.3 to 4.6 h per cycle).
- Cycle 4: switch-over from the check to the test condition, logged as one long cycle (about double the Ah). Not a capacity value; excluded.
- Then the test condition, with a new block of check cycles roughly every 500 cycles (sometimes every 130 to 260).
- For partial-DoD cells a normal cycle moves only part of the charge (for example 0.57 Ah of 1.07 Ah for an LFP 20-80 % cell), so normal cycles are not capacity.
- First check, median by chemistry: LFP 1.051 Ah, window 2.0 to 3.6 V; NMC 2.951 Ah, 2.0 to 4.2 V; NCA 3.122 Ah, 2.5 to 4.2 V.

SoH (`src/sandia_soh.py`, log `outputs/sandia_soh_run.log`)
- Check cycle = full voltage window (within 0.05 V of the first checks) and cycle duration at least 90% of the first checks. Consecutive check cycles (gap <= 10) form a block; block capacity = median discharge Ah. SoH = block capacity / first block capacity.
- Aging axes saved per block: cycle number, equivalent full cycles (cumulative discharged Ah / first capacity), days since start.
- Excluded: SNL_18650_NMC_25C_20-80_0.5/3C_b (data in the archive starts at cycle 414, so no first check and no reference capacity).
- Dropped as end-of-test artifacts (rule: last check of a cell with 3+ checks falling more than 0.25 in one step): NCA 15C 0-100 2C _a (0.668 to 0.026, a near-empty check) and NMC 25C 40-60 3C _a (0.893 to 0.625, one cycle after a 4,000-cycle gap). The rule is not applied to cells with only 2 checks, because a single large drop there can be real (all four NMC 15 C cells fall to 0.74 to 0.79 by cycle 260).
- Kept: 5 small rises between checks (for example 3 LFP 3C cells at cycle 3051, likely recovery after a test pause).
- Result: 85 cells, 651 SoH points, SoH 0.542 to 1.000. Checks per cell: min 2, median 6, max 25. 7 cells have only 2 checks.
- Final SoH, median by chemistry and condition, in `outputs/sandia_cells.csv`. Pattern: LFP fades slowly (SoH 0.83 to 0.93 after about 1,900 to 3,900 equivalent full cycles); NMC and NCA fade much faster per cycle (NMC 0.76 to 0.93, NCA 0.58 to 0.90, mostly within 200 to 1,700 equivalent full cycles). Higher discharge rate and 35 C fade faster; 15 C is also bad for NMC and NCA.
- Figure: `figures/sandia_soh.png` (SoH vs equivalent full cycles, coloured by temperature, line style by DoD).

Notes for the next step
- SoH labels are sparse (one per check block), so Model 1 on Sandia has 651 points from 85 cells. Testing must hold out whole cells or whole conditions.
- Sandia SoH goes down to 0.54 (NCA). The Severson R(SoH) curve is only backed by data down to 0.76, so for low SoH it is an extrapolation, and it is an LFP curve.

## Model 1 on Sandia (24 Oct 2026)

Scripts: `src/sandia_model1.py` (with the input "days"), `src/sandia_model1_nodays.py` (main model, without "days"). Logs `outputs/sandia_m1_run.log`, `outputs/sandia_m1_nodays_run.log`.

Setup
- Target: SoH at each capacity check. The first check of every cell (SoH = 1 by definition) is not used: 566 points, 85 cells, 33 conditions (LFP 328 points, NMC 152, NCA 86).
- Inputs (operating conditions and usage only): chemistry (one-hot), temperature (C), DoD (%), discharge rate (C), nominal equivalent full cycles = cycle number x DoD / 100 (from the test plan, not from measured Ah). Charge rate is 0.5C for all cells, so it is not an input.
- Not used, to avoid leaking the answer: capacity, energy, mean voltages, efficiencies, measured Ah throughput.
- "days since start" was tested as an extra input and dropped: at a fixed current a cell with less capacity finishes each cycle sooner, so days could encode capacity. Effect was small (forest, unseen conditions: R2 0.815 with days, 0.805 without; unseen cells 0.875 vs 0.874), so there was no real leak, but the main model is the one without days.
- Tests, all scores out-of-fold: (1) unseen cells: GroupKFold(5) by cell; (2) unseen conditions: leave one condition out (33 folds; all replicate cells of that condition unseen). Light tuning inside each training fold with GroupKFold(3) by cell.
- Baselines: training mean; a straight line of SoH vs nominal equivalent full cycles per chemistry (fitted on the training fold).

Results, main model (without days), out-of-fold:

| model | unseen cells R2 / RMSE | unseen conditions R2 / RMSE |
|---|---|---|
| baseline: training mean | -0.005 / 0.065 | -0.048 / 0.067 |
| baseline: line per chemistry | 0.551 / 0.044 | 0.496 / 0.046 |
| linear regression | 0.579 / 0.042 | 0.500 / 0.046 |
| decision tree | 0.825 / 0.027 | 0.583 / 0.042 |
| **random forest** | **0.874 / 0.023** | **0.805 / 0.029** |
| gradient boosting | 0.870 / 0.024 | 0.766 / 0.032 |

(Baselines and with-days numbers in `outputs/sandia_m1_metrics.csv`; without-days in `outputs/sandia_m1_metrics_nodays.csv`.)

- Random forest RMSE by chemistry, unseen conditions: LFP 0.019, NMC 0.033, NCA 0.047.
- With days, by SoH range (unseen conditions, forest): SoH >= 0.9 RMSE 0.017; 0.8 to 0.9: 0.034; below 0.8: 0.069. Low-SoH points are few (mostly NCA) and hardest.
- Worst unseen conditions (with days, forest, MAE): NMC 15 C 2C (0.080) and 1C (0.059), NCA 15 C 1C (0.057), NMC 35 C 2C (0.050), NCA 15 C 2C (0.047). Cold (15 C) cycling of NMC/NCA fades much faster, and when a 15 C condition is held out the model has seen few other cold examples.
- `figures/sandia_m1_trajectories.png`: on unseen conditions the forest follows the trend well where similar conditions exist (for example NMC 25 C 20% DoD, NCA 25 C 2C), but cannot extrapolate beyond what it has seen: for LFP 25 C 3C it levels off at about 0.91 while the cells reach 0.83 to 0.84.
- Input importance (permutation, final forest with days, on the training data, indicative only): chemistry (LFP flag) and equivalent full cycles matter most, then DoD, discharge rate, temperature.
- Saved models: `outputs/models/sandia_m1_nodays__{linear,tree,forest,gboost}.joblib` (main), `outputs/models/sandia_m1__*.joblib` (with days). Figures: `figures/sandia_m1_pred_vs_actual.png`, `figures/sandia_m1_trajectories.png`, `figures/sandia_m1_importance.png` (these three come from the with-days run; the without-days numbers are within 0.01 in R2).

Limitations
- 18650 lab cells, constant-current cycling, not grid-size cells or a real grid duty profile.
- 33 conditions only; within one chemistry, some combinations are missing (for example no 15 C or 35 C at partial DoD), so an unseen condition is often an extrapolation.
- Tree models do not extrapolate past the cycle counts and SoH values seen in training.

## Full chain per chemistry (24 Oct 2026)

Script `src/chain_sim.py`, log `outputs/chain_run.log`, tables `outputs/chain_soh.csv`, `outputs/chain_results.csv`.

Chain: grid duty -> Sandia Model 1 (SoH) -> Severson quadratic R(SoH) -> P_max fraction -> real Grid-India hours.

Design (all assumptions)
1. Duty: 1 equivalent full cycle per day (365 per year). Three scenarios, each a condition Sandia tested for all three chemistries (4 cells each): main 25 C / DoD 60% / 0.5C; harder 25 C / DoD 100% / 1C; hot 35 C / DoD 100% / 1C.
2. SoH per year from the Sandia Model 1 random forest without "days". Year 0 = 1 by definition. Pessimistic case = SoH minus the chemistry's out-of-fold RMSE on unseen conditions (LFP 0.019, NMC 0.033, NCA 0.047).
3. Years beyond the longest Sandia test of that condition are flagged "beyond_test". Sandia covered (at 365 per year): main LFP 13.8 years, NMC 6.3, NCA 2.3; harder LFP 9.7, NMC 1.8, NCA 1.4; hot LFP 9.7, NMC 1.8, NCA 1.8. Beyond that the forest holds SoH flat (it cannot extrapolate), so later values are a lower bound on aging, not a prediction that aging stops.
4. R(SoH): Severson quadratic for all chemistries. For NMC and NCA this is an assumption (LFP curve), so differences between chemistries here come only from how fast SoH falls. Literature +30% and +100% curves run as sensitivity. SoH < 0.76 flagged.
5. Grid: 8,468 clean Grid-India hours from 1 Jul 2024 to 24 Jun 2025. Battery sized against the lowest-inertia hour in that window (2 Jun 2025 12:00, H = 3.02 s, peak request 0.0398 pu for a 10% loss), demand ratio 0.8 / 1.0 / 1.2. 10% loss only. Energy reserve not limited in this run (earlier runs showed a 10 s reserve never bound).

Results (quadratic R, demand ratio 1.0, values within the Sandia test length unless marked)

| scenario, year | chemistry | SoH | P_max / fresh | % of hours where aging matters |
|---|---|---|---|---|
| main, year 2 | LFP | 0.975 | 0.988 | 0.0 |
| main, year 2 | NMC | 0.878 | 0.915 | 2.0 |
| main, year 2 | NCA | 0.837 | 0.874 | 10.9 |
| main, year 10 | LFP | 0.917 | 0.949 | 0.2 |
| main, year 6 | NMC | 0.836 | 0.873 | 10.9 |
| harder, year 1 | LFP / NMC / NCA | 0.982 / 0.837 / 0.857 | 0.992 / 0.874 / 0.894 | 0.0 / 10.9 / 5.3 |
| hot, year 1 | LFP / NMC / NCA | 0.967 / 0.843 / 0.838 | 0.984 / 0.880 / 0.875 | 0.0 / 9.9 / 10.9 |
| hot, year 9 | LFP | 0.863 | 0.900 | 4.4 |

- Largest RoCoF increase vs a fresh battery in any hour (within test): LFP main year 10: +0.005 Hz/s; NCA main year 2: +0.026 Hz/s; NCA hot year 2 (just beyond test): +0.061 Hz/s.
- Pessimistic SoH, main scenario year 10: LFP 0.7% of hours, NMC 23.0% and NCA 30.9% (both beyond test).
- Literature +100% resistance: NMC and NCA matter in 100% of hours from year 1; LFP reaches 33% of hours by year 5 (main) and 100% by year 11. Literature +30%, main year 10: LFP 6.3%, NMC 31.6%, NCA 34.1%.
- Figures: `figures/chain_soh_years.png` (SoH per year, band = pessimistic, open markers beyond test), `figures/chain_hours_years.png` (% of hours where aging matters), `figures/chain_pmax_rocof_years.png` (power ceiling and midday RoCoF per year, main scenario).

What this says
- With a resistance rise like our LFP data, an LFP grid battery keeps its synthetic inertia almost untouched for about a decade of daily cycling (SoH about 0.92 after 10 years in the main scenario, aging matters in under 0.3% of hours). A hot site (35 C, full DoD, 1C) brings this to about 4% of hours by year 9.
- NMC and NCA lose capacity much faster under the same duty, so the same resistance curve gives a lower power ceiling within 1 to 2 years, and aging matters in about 10% to 40% of the latest year's hours.
- The chemistry ranking (LFP clearly best for long inertia duty) holds for every scenario and every R(SoH) curve tested. The size of the effect depends strongly on the resistance curve and on battery sizing.

Limitations of the chain
- Model 1 cannot extrapolate past the tested cycle counts (NMC and NCA tests are short), so long-term NMC/NCA results are lower bounds.
- One R(SoH) curve (LFP) for all chemistries; chemistry-specific resistance growth is not known from our data.
- Duty of 1 cycle per day and the three conditions are assumptions; real grid batteries see partial, irregular cycling and calendar aging.
- Lab 18650 cells, not grid-size cells. Grid model limitations from the real-grid run still apply (governor too strong, nadir optimistic).

## Grid calibration data found (24 Oct 2026)

Source: POSOCO (now Grid-India) with IIT Bombay, "Assessment of Inertia in Indian Power System", January 2022, 124 pages. Downloaded from the Prayas Energy Group archive (https://archive.prayaspune.org/collection/5/document/1167) to `data/grid_events/posoco_inertia_2022.pdf`; text in `posoco_inertia_2022.txt`. Extracted numbers with page references in `data/grid_events/calibration_targets.csv`. (The NASPI 2023 slides link now returns 404; the report replaces them.)

Confirmed from the full report (earlier these were only from slides):
- System inertia 5 to 9 s (2014 to 2021). POSOCO's own frequency model, tuned on 26 events (Mar 2020 to Nov 2021), uses an average H of 7 s on demand base.
- Average power number (loss / drop at nadir) 10,000 MW/Hz; median steady-state frequency response 15,000 MW/Hz (up from 4,000 MW/Hz in 2015).
- Time to nadir 9 to 14 s (average about 10 s); frequency settles in about 50 s; primary response starts within 2 to 4 s.
- Load damping 2 to 5 % per Hz.
- POSOCO model: first-order governor 0.8 s and turbine 10 s, 5 % droop.
- Reference contingency for defence plans: 4,500 MW (simulated nadir 49.55 Hz).
- One fully documented real event: 21 Jan 2019 15:57, 780 MW lost out of 144,482 MW generation, initial RoCoF 0.0167 Hz/s, nadir 49.804 Hz (drop 0.081 Hz) after 10.6 s, estimated inertia 8.06 s.
- Largest observed event: 28 May 2020, 5,346 MW lost, 50.02 to 49.54 Hz (time and demand not given in the text).

What this means for our grid model (comparison, not yet changed):
- Governor: ours is about 70,000 to 80,000 MW/Hz and reaches the nadir in about 3 s; real is about 15,000 MW/Hz steady state and 9 to 14 s. Confirms the earlier concern.
- Inertia: our rule H = 5 s x synchronous share gives 3 to 5 s, below the 5 to 9 s (average 7 s) found by POSOCO. A quick check: for the 21 Jan 2019 event, RoCoF = loss / (2H) x 50 Hz gives H = (780 / 144,482) x 50 / (2 x 0.0167) = 8.1 s, matching POSOCO's 8.06 s. So our inertia is too low and our RoCoF values are too high (pessimistic), not only the nadir being optimistic.
- Event size: our 5 % and 10 % losses (about 10 and 20 GW) are far above real events (largest 5.3 GW; reference contingency 4.5 GW).

## Grid model calibrated to the Indian grid, and results rerun (24 Oct 2026)

These results REPLACE the real-grid and chain results above (which used the uncalibrated Task 3 grid). Scripts: `src/grid_cal.py` (model and calibration), `src/real_grid_cal.py` (real hours and chain), `src/compare_calibration.py`. Logs: `outputs/grid_cal_run.log`, `outputs/real_grid_cal_run.log` (Part B, chain), `outputs/real_grid_cal_run_partA.log` (Part A, final).

Model changes
- Two-stage governor (0.8 s) and turbine (10 s), POSOCO's own model structure.
- H = 9.25 s x synchronous share; governor gain = 4.25 x synchronous share; load damping D = 1.5 pu/pu (3 % per Hz, inside POSOCO's 2-5 %). Fitted by grid search to the 21 Jan 2019 event plus a soft target of 15,000 MW/Hz steady response. Synchronous share at that event is not in our data (starts Sep 2021); proxy = median of January 15:00-16:59 hours in 2022 = 0.891.
- Events: 4,500 MW (IEGC reference contingency) and 5,346 MW (largest observed, 28 May 2020), as a share of each hour's real demand.
- Battery: fixed size in MW. Sized against the lowest-inertia clean hour (2 Jun 2025 11:00, synchronous share 0.605, demand 205 GW, H = 5.55 s): peak inertia request for 4,500 MW = Hv / (H + Hv) x 4,500 = 1,192 MW; fresh P_max = 1,192 MW / d, d = 0.8, 1.0, 1.2. Hv = 2 s, energy reserve 10 s x SoH, unchanged.

Calibration check (no battery)

| check | measured / POSOCO | calibrated model | old model |
|---|---|---|---|
| 21 Jan 2019, 780 MW: initial RoCoF | 0.0167 Hz/s | 0.0164 | about 1.7 x faster |
| same: drop to nadir | 0.081 Hz | 0.083 Hz | |
| same: time to nadir | 10.6 s | 10.7 s | about 3 s |
| steady response (that hour) | about 15,000 MW/Hz (median) | 15,277 MW/Hz | 70,000 to 80,000 |
| 4,500 MW reference, nadir / quasi-steady | 49.55 / 49.71 Hz (POSOCO simulation) | 49.56 / 49.73 Hz (2021 median hour) | |
| 28 May 2020, 5,346 MW, drop | 0.48 Hz (demand not reported) | 0.47 Hz at 170 GW (0.53 at 150, 0.42 at 190) | |
| time to nadir | 9-14 s | 10.8 s | |

Figure: `figures/grid_cal_check.png`. Calibrated H over the data: median 8.4 to 8.8 s per year, lowest hour 5.6 s (2025), so inside or near POSOCO's 5-9 s.

Two bugs found and fixed while running this step
1. Grid cleaning: 7 partial hours next to SCADA outages had passed (for example 3 Jul 2022 12:00 with 13,701 MW demand), because blank hours were inside the rolling median. Fix in `src/clean_grid_data.py`: blank hours left out of the median, plus a new rule dropping hours below 60 % of the month's median demand. Now 747 hours dropped, 32,685 kept; lowest demand 105,322 MW (night after Diwali 2021). The 5 remaining hours below 70 % of the month median are consecutive night hours on festival nights or winter nights and were kept. The old uncalibrated real-grid and chain outputs were made with the older cleaning (6 extra hours; their sizing hour was not affected) and are superseded anyway.
2. In the first version of `real_grid_cal.py` the battery was a fixed share of demand and the "no battery" case was simulated as an unlimited battery. Fixed: battery fixed in MW (per-unit cap = MW / that hour's demand), and "no battery" simulated with the battery switched off.

Results, all 32,685 clean hours (Sep 2021 to Jun 2025)
- 4,500 MW loss, no battery: nadir 49.382 to 49.713 Hz, RoCoF 0.055 to 0.120 Hz/s. Fresh battery (d = 1.0): nadir 49.417 to 49.730 Hz, RoCoF 0.044 to 0.094 Hz/s. The old model gave RoCoF 0.48 to 0.77 Hz/s for a 10 % loss, so earlier RoCoF values were much too high.
- Load shedding (first stage 49.4 Hz): 4,500 MW: 6 hours below 49.4 Hz without battery, 0 with a fresh or aged battery (SoH 0.9, 0.82, 0.7). 5,346 MW: 925 hours (2.83 %) without battery, 417 (1.28 %) with a fresh battery, 417 at SoH 0.9 and 0.82, 419 at SoH 0.7. So the battery halves the load-shedding hours, and aging removes almost none of that benefit.
- Lowest nadirs come from low-demand nights (a fixed 4,500 MW is a bigger share), while the hours where the battery hits its cap are the low-inertia sunny middays. Two different "hard" hours.
- % of hours where aging matters (nadir or RoCoF changes by more than 0.001 vs fresh), 4,500 MW, quadratic curve:

| battery size d | SoH | 2021 | 2022 | 2023 | 2024 | 2025 | all hours | before calibration (all hours) |
|---|---|---|---|---|---|---|---|---|
| 0.8 | 0.82 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.0 | 0.9 | 0 | 0 | 0 | 0 | 0.05 | 0.01 | 0.19 |
| 1.0 | 0.82 | 0 | 0 | 0 | 0.09 | 3.8 | 0.5 | 5.4 |
| 1.0 | 0.7 (extrap.) | 9.2 | 16.2 | 22.7 | 27.0 | 33.5 | 22.3 | 70.5 |
| 1.2 | 0.82 | 12.9 | 19.7 | 25.1 | 28.4 | 34.6 | 24.7 | 79.7 |

- Size of the effect is small: largest RoCoF increase vs fresh at SoH 0.82, d = 1.0: 0.003 Hz/s (on about 0.09 Hz/s, about 3 %); largest nadir change 0.0001 Hz. Literature +100 % curve at SoH 0.82: up to 0.011 Hz/s and 0.0025 Hz. Before calibration these were about 10 times larger (0.032 Hz/s for the quadratic case).
- 5,346 MW event, d = 1.0, SoH 0.82: aging matters in 24.7 % of hours (the battery is sized for 4,500 MW, so it is undersized for this event).
- Hardest hour trace (`figures/real_grid_cal_trace_hardest_hour.png`): battery cap 1,192 MW fresh, about 1,020 MW at SoH 0.82 (clips for about 1.5 s), 627 MW with the literature +100 % curve (clips for about 4.5 s). Frequency curves of fresh and SoH 0.82 almost overlap; the battery lifts the nadir from about 49.58 to 49.61 Hz in that hour.
- Expectation before running was that the aging effect on the nadir would grow; it did not. With realistic (higher) inertia and real event sizes the inertia request is small and short, and the nadir is set by the governors about 10 s later. Pure synthetic inertia mainly affects RoCoF.
- Energy reserve (10 s) was never used up.

Chain per chemistry (latest 12 months, 4,500 MW, d = 1.0, quadratic): % of hours where aging matters
- Main (25 C, 60 % DoD, 0.5C): LFP 0 % through year 12; NMC 0.05 % (year 2), about 0.9 to 1.2 % after; NCA 0.9 % (year 2), about 1.4 to 1.5 % after.
- Harder (25 C, 100 %, 1C): LFP 0 %; NMC 0.9 % (year 1), 2.1 % after; NCA 0.2 % (year 1), 14 to 15 % after year 2 (beyond Sandia test length from year 2).
- Hot (35 C, 100 %, 1C): LFP 0 % until year 5, 0.18 % at year 9; NMC 0.4 % (year 1), about 3 % after; NCA 0.9 % (year 1), about 22 % after year 2.
- Literature +100 % curve: NCA and NMC 100 % of hours from year 2 to 3; LFP 12 % at year 5 and 36 % at year 12 (main).
- Figure `figures/chain_cal_hours_years.png`; table `outputs/chain_cal_results.csv`.

Other outputs: `outputs/grid_cal_params.json`, `outputs/grid_cal_fit.csv`, `outputs/real_grid_cal_summary.csv`, `outputs/real_grid_cal_hourly.csv`, `outputs/real_grid_cal_cases.csv`, `outputs/calibration_before_after.csv`; figures `real_grid_cal_aging_hours_by_year.png`, `real_grid_cal_2025_by_hour.png`, `calibration_before_after.png`.

Remaining limitations
- Calibrated to one fully documented event plus POSOCO summary figures; the synchronous share for that 2019 event is a proxy.
- Single-area model; governor deadbands and the spread of response across plants are not modelled; POSOCO notes its own 5 % droop assumption is idealised.
- Battery is pure virtual inertia (Hv = 2 s); a battery giving fast frequency response would interact with aging differently.

## MAIN STUDY: solar-only and wind-only systems, battery as the only inertia (24 Oct 2026)

Agreed with the user: the main study is a system powered ONLY by solar (or ONLY by wind), where the grid-forming battery is the only source of inertia and frequency support. The national-grid work above becomes context (to be decided when writing the report).

Scripts: `src/review_kutch.py` (data review), `src/re_only.py` (duty, aging, events), `src/plot_re_only.py` (figures). Logs: `outputs/re_only_run.log`, `outputs/re_events_run.log`. Tables: `outputs/re_duty.csv`, `re_soh.csv`, `re_events.csv`, `re_events_by_hour.csv`, `re_dispatch_fresh.csv`.

Data (real): Kutch, Gujarat (23.25 N, 69.67 E), 2025, Renewables.ninja MERRA-2, hourly solar and wind output per kW and air temperature. Review: solar produces in 3,869 h (44 %, daytime only), wind in 7,362 h (84 %, strongest 18-23 h); largest hour-to-hour drop 22 % (solar, 99th pct) and 14 % (wind); longest near-zero spell 16 h (solar, nightly) and 51 h (wind); air temperature at high output 32.2 C (solar) vs 28.1 C (wind).

System (assumptions)
- 100 MW plant + grid-forming battery 25 MW / 50 MWh; round-trip efficiency 90 %. Load = plant output + battery dispatch in each hour.
- The battery keeps a reserve for frequency support: 12.5 MW of power (design trip 10 MW + 25 % margin) and the charge band 10-30 % (dispatch never goes below 30 %).
- The rest is used to firm the plant towards its 24-hour centred average output (dispatch +/- 12.5 MW, SOC 30-90 %).
- Two operating modes as the battery ages: SoH-unaware (dispatch limit stays 12.5 MW, so aging eats the reserve) and SoH-aware (dispatch limit = power ceiling(SoH) - 12.5 MW, reserve protected).
- Events in every hour the plant produces >= 1 MW: design event = sudden loss of 10 MW (10 % of plant), stress event = 25 MW (25 %, equal to the battery rating), both capped at what the plant produces.
- Covered = spare power (ceiling - dispatch) >= loss AND charge above 10 % can supply (dispatch + loss) for 15 min. If not covered, frequency falls until load damping (D = 1.5 pu/pu, 3 %/Hz) covers the gap: drop = shortfall / (D x load) x 50 Hz; load shedding if the drop > 0.6 Hz (49.4 Hz).
- Virtual inertia Hv = 4 s on the 25 MW rating: initial RoCoF = 2.5 Hz/s (10 MW loss) and 6.25 Hz/s (25 MW), the same fresh or aged as long as the battery covers the loss (it is a control setting).
- This first version replaced an earlier attempt without a reserve, in which even a fresh battery failed 14-49 % of events (battery already busy shifting energy) and the 25 % trip equalled the full rating; that was a design flaw, not an aging result.

Battery duty (fresh, from the real hourly output)
- Solar: 208 equivalent full cycles per year, daily depth of discharge 60 % (full dispatch window), discharge rate 0.18C, 28.0 C during use. Pattern: charges 08-10 h, sits full (90 %) all midday, discharges 17-20 h, at the 30 % floor at night.
- Wind: 214 cycles per year, depth 60 %, 0.10C, 26.5 C. Pattern: charges in the evening, discharges in the morning, SOC 45-70 % most of the day.
- Energy shifted per year: about 10,400 MWh (solar) and 10,700 MWh (wind).
- Model 1 inputs: rates below Sandia's lowest (0.5C) set to 0.5C; temperature 28.0 / 26.5 C (sensitivity 35 C).

Aging (Sandia Model 1): solar and wind batteries age almost the same, because the duty is almost the same (the battery is small next to the plant, so both partners use the full window daily).
- SoH after 5 / 10 / 15 years: LFP 0.966 / 0.945 / 0.926; NMC 0.853 / 0.836 / 0.837; NCA 0.826 / 0.823 / 0.823 (solar; wind within 0.01).
- Beyond Sandia test length: NCA from year 4-5, NMC from year 11-12 (SoH held flat after that, a lower bound on aging).
- Power ceiling (quadratic R): LFP 24.3 MW at year 10; NMC 21.8 MW; NCA 21.5 MW. NMC and NCA fall below 22.5 MW (reserve 12.5 + design trip 10) in years 3-4.

Results (site temperature, quadratic R(SoH))
- Design event (10 MW), SoH-UNAWARE, % of event hours not covered: LFP 0 % in all 15 years (solar and wind). NMC: solar 0.45 %, wind 1.14 % (year 10). NCA: solar 0.59 %, wind 1.24 % (year 10). Load shedding at year 10: NMC 0.26 % (solar) / 0.64 % (wind); NCA 0.45 % / 0.97 %.
- Design event, SoH-AWARE: 0 % not covered for every chemistry, year and source. Cost in energy shifting: none for solar, at most 0.15 % for wind (for example NCA year 10: 8,968 vs 8,981 MWh). Knowing SoH and protecting the reserve removes the aging problem for the design event at almost no cost.
- When it fails (`re_fail_by_hour.png`): wind around 07-09 h (battery discharging in the morning while wind is still producing), solar around 17 h (battery discharging in the evening ramp while the plant still produces). The aged battery fails when it is already busy.
- Wind is about twice as exposed as solar: twice as many producing hours (8,063 vs 4,228 event hours) and the battery is more often discharging when events happen.
- Stress event (25 MW = full rating): fresh battery 0 % not covered (solar) and 14 % (wind, morning discharge). Any aging leaves the battery short when idle at high output: 47-58 % not covered from year 1. Load shedding: LFP solar 0 % to year 5, 9.6 % at year 10, 25 % at year 15; LFP wind 12-13 % to year 5, 26 % at 10, 43 % at 15; NMC and NCA 49-57 % from year 2-4. LFP shortfalls are small (drop below 0.6 Hz); NMC/NCA shortfalls are large.
- R(SoH) sensitivity, design event, SoH-unaware, year 10, % not covered: literature +30 %: NMC/NCA 1.4-2.1 %, LFP 0 %; literature +100 %: LFP 1.7 % (solar) / 2.7 % (wind), NMC/NCA 8.8-10.5 %.
- Hot 35 C (no cooling), year 10: slightly worse (NCA solar 0.83 % vs 0.59 %).

Sizing rule (from the curves): to keep the design event covered at end of life without SoH-aware operation, the power reserve must be about design trip x R/R0 at end of life plus the dispatch limit. With our curve: LFP needs about 1.06 x (SoH 0.93), NMC/NCA about 1.17-1.19 x (SoH 0.82-0.84), i.e. 6 % vs 17-19 % extra power. CORRECTION (25 Oct): recomputed from the saved curve at the lowest SoH over 15 years (site temperature), the rating needed to keep 25 MW deliverable is LFP 1.046-1.048 x, NMC 1.149-1.150 x, NCA 1.165 x, i.e. +5 %, +15 %, +17 %. These are the numbers used in the report.

Figures: `re_average_day.png`, `re_soh_power_years.png`, `re_events_years.png`, `re_fail_by_hour.png`.

Limitations
- Isolated 100 MW plant + battery with assumed sizes, reserve, dispatch rule and load; only the solar/wind output and temperature are real data. No real event to calibrate against.
- Hourly data; the events are defined tests, not measured. Pass/fail plus a damping-based frequency drop is a simplification of the grid-forming inverter's behaviour at its current limit.
- Model 1 cannot extrapolate past Sandia's test length; discharge rates below 0.5C are set to 0.5C; temperature differences of 1-2 C are not resolved by the model.
- LFP resistance curve used for all chemistries.

## Battery and reserve sizing sweep (24 Oct 2026)

Why: the abstract asks "whether battery selection or energy buffer sizing can help sustain adequate grid-support performance as the battery ages". The main study used one size only.

Script `src/re_sizing.py` (same chain as `re_only.py`: duty of the fresh battery -> Sandia Model 1 SoH -> quadratic R(SoH) power ceiling -> aged dispatch -> design and stress events; site temperature, quadratic curve). Output `outputs/re_sizing.csv`, log `outputs/re_sizing_run.log`, figures `re_sizing_design.png`, `re_sizing_cost_stress.png` (`src/plot_re_sizing.py`).

Sizes (kept realistic for a 100 MW plant, as agreed with the user): power 15 / 25 / 35 MW; duration 1 / 2 / 4 h (energy = power x duration, 15 to 140 MWh); power reserve 10 / 12.5 / 15 MW (design trip + 0 / 25 / 50 % margin). Combinations with reserve >= power skipped: 24 sizes.

Change to the SoH-aware rule (found by this sweep): the first SoH-aware rule protected only the power reserve. For 1 h batteries the aged battery then failed on ENERGY (charge between 10 and 30 % of the shrunken capacity could not hold the trip for 15 min), and SoH-aware was worse than unaware (15 MW / 1 h, NCA solar 15 % vs 6.6 %). Now SoH-aware also raises the lowest dispatch charge so that (dispatch limit + power reserve) can be held for 15 min with the aged capacity (never below 30 %): `aware_soc_min()` in `re_only.py`. Rerun of the main study: every number identical (for 25 MW / 50 MWh the 30 % floor was already enough).

Results, year 10 unless stated
- Fresh battery: design event covered in 100 % of hours for every size.
- Design event, SoH-unaware: the power margin decides it. With 35 MW nothing fails for solar and at most 1.1 % for wind (NCA, 4 h). With 25 MW: 0 to 1 % solar, 0 to 4 % wind (as in the main study). With 15 MW and reserve 10 MW (no margin) even LFP fails (3 % solar, 2.6 to 14.5 % wind); NMC/NCA up to 11 % (solar) and 31 % (wind, 1 h). A 15 MW battery with 12.5 MW reserve covers it at 2 to 4 h, but at 1 h NMC/NCA fail 6.5 % (solar) and 24 % (wind) on energy.
- A reserve of 15 MW (50 % margin) gives 0 % for every chemistry and size; it is the reserve size that absorbs the NMC/NCA power fade (about 15 % at year 10).
- More energy is NOT always better: for wind, longer batteries fail more (25 MW, NCA: 0.27 % at 1 h, 1.24 % at 2 h, 4.14 % at 4 h). With more energy the battery shifts more (4,791 vs 8,981 vs 16,507 MWh/yr) and is more often discharging at its full limit when a trip comes, so less spare power. All of these are power failures, not energy failures.
- Design event, SoH-aware: 0 % not covered for every size, chemistry and source. Its price (energy shifting given up): about 0 for 2 and 4 h batteries of 25 and 35 MW (at most 5 %, wind NCA 25 MW / 4 h / reserve 15); about 11 to 12.5 % for every 1 h battery even LFP (it must keep more charge back); 33 to 78 % for NMC/NCA at 15 MW with reserve 12.5 MW (after aging almost no power is left for shifting, the battery becomes a frequency-only battery).
- Stress event (25 MW trip), 2 h, reserve 12.5, load shedding: 15 MW: 77 to 84 % for every chemistry; 25 MW: LFP 10 % (solar) / 26 % (wind), NMC/NCA 55 to 57 %; 35 MW: 0 % solar, 2 to 5 % wind. The stress trip needs a battery larger than the trip; then chemistry matters much less.
- Aging barely changes with size: daily depth stays at the full 60 % window in all sizes except 15 MW / 4 h / reserve 12.5 MW (only 2.5 MW for shifting: DoD 36 to 39 %, 121 to 136 cycles/yr, year-10 SoH LFP 0.97, NMC 0.885, NCA 0.863 vs 0.945 / 0.836 / 0.823). Wind cycles vary with size (179 to 238 per year; solar always 208, one cycle per day), so with 4 h batteries wind ages slightly slower early on (NMC year 4: 0.874 to 0.878 vs 0.865 for solar), but by year 10 Model 1 is flat (beyond the Sandia test length for NCA and NMC at these cycle counts) and the difference disappears. So "solar and wind age the same" holds for the realistic sizes here, not as a general rule.

What this says for the abstract question
- Battery selection: LFP keeps the design event covered without extra margin in every size with reserve >= 12.5 MW; NMC/NCA need either a 15 MW reserve or SoH-aware operation.
- Buffer sizing: the POWER margin (reserve over the design trip) matters most; more energy helps only against energy failures of 1 h batteries and can even hurt wind by keeping the battery busy. Battery power must exceed the largest trip to cover it at end of life.

## Frequency response over time, inverter-only system (24 Oct 2026)

Why: the abstract asks for "the achievable synthetic inertia response". The main study only checked pass/fail plus an end-point frequency drop. This run simulates the first 20 s after each trip.

Script `src/re_freq.py`, figure `src/plot_re_freq.py` -> `figures/re_freq_example.png`. Outputs `outputs/re_freq_summary.csv` (all producing hours), `re_freq_example_hours.csv`, `re_freq_traces.csv`, log `re_freq_run.log`.

Model (assumptions marked)
- Same system, real hours and battery states as the main study (25 MW / 50 MWh, reserve 12.5 MW, year 10 with site temperature, quadratic R(SoH); fresh = year 0).
- Grid-forming battery as a virtual synchronous machine: asks for p = -2 Hv S dw/dt - Kd S w - Ki S (integral of w). Hv = 4 s (as before); droop 1 % on the battery rating (Kd = 100, ASSUMED); restoring control Ki = Kd / 10 s, so frequency returns to 50 Hz and the battery ends up carrying the whole loss (ASSUMED, matches the main study's "covered" test).
- It can add only its spare power (ceiling - dispatch). At that limit its inertia and droop are gone, and frequency is held only by the load: motor inertia H_L = 0.5 s on the load (ASSUMED; sensitivity 0.2 and 1.0 s) and load damping D = 1.5. Load = plant + dispatch, as before. Load shedding is flagged at 49.4 Hz, not modelled as an action.

Results, year 10, all producing hours
- Consistency check: % of hours below 49.4 Hz is exactly the main study's load-shedding share (design event: NMC 0.26 % solar / 0.65 % wind, NCA 0.45 % / 0.97 %), and % of hours at the power limit equals its "not covered" share. Expected: the end point of the time model is the same balance as the main study's formula. The time model adds speed and recovery.
- Battery within its limit (fresh, LFP, every SoH-aware case, design event): frequency dips to 49.81 Hz in about 0.4 s, RoCoF 2.0 Hz/s at the first instant and 0.37 Hz/s over 500 ms, and recovers to 49.97 Hz by 20 s. Aging does not change this at all, because inertia and droop are control settings.
- Battery at its limit (aged NMC/NCA, SoH-unaware): frequency keeps falling and does NOT recover (the restoring control has no power left). Worst hours: 49.0 Hz (NMC) and 48.5 Hz (NCA). 500 ms RoCoF up to 1.0 (NMC) and 1.6 Hz/s (NCA), versus 0.37 within limit.
- Example hours (`re_freq_example.png`): solar 20 Mar 2025 17:30 and wind 27 Mar 2025 11:30, where the aged NCA falls short. Design trip: fresh / LFP / NCA SoH-aware dip to 49.81 Hz and recover; NMC settles at 49.60 Hz (solar) and 49.17 Hz (wind); NCA at 49.11 / 48.74 Hz, crossing 49.4 Hz in under 1 s.
- Stress trip (25 MW): solar median 49.54 Hz when covered; hours at the limit go far lower (worst 45 to 46 Hz solar, 36 to 39 Hz wind, even for the fresh battery in wind hours where the battery is busy). In reality under-frequency relays and plant protection would act long before that, so these values only mean "system collapses" and are outside what this model is valid for.
- H_L sensitivity (0.2 / 0.5 / 1.0 s): no change in which hours fall below 49.4 Hz or how low; only the speed of the fall in failed hours (500 ms RoCoF NCA solar 2.55 / 1.60 / 0.98 Hz/s).

What this adds for the report
- Within its power limit, an aged battery gives exactly the same inertia response as a fresh one. Aging matters only when it pushes the battery to its power limit, and then the response does not degrade gently: the battery stops controlling frequency and the frequency falls to where the load alone can balance, without recovery.
- This supports the main-study metric (share of hours where the aged battery hits its limit) as the right measure of aging impact.

## Report draft (25 Oct 2026)

Agreed with the user: 2-page report built on the solar-only / wind-only study. Kept: Sandia Model 1, Model 2 (resistance, with the model comparison and why the quadratic is used), Kutch scenario, frequency simulation, SoH-unaware vs SoH-aware, one sizing paragraph. Left out: Severson Model 1, Task 3 generic simulation, national grid (Grid-India, POSOCO calibration), curve sensitivities, hot 35 C, stress event, process history. Limitations written as future scope.

Files: `report/main.tex` (IEEEtran conference, two columns; no LaTeX compiler on this machine, compile on Overleaf), figures `report/fig_A.png`, `fig_B.png`, `fig_C.png` from `src/plot_report.py` (PNG at 300 dpi, because a Windows Application Control policy blocked matplotlib's PDF backend and parts of scikit-learn on 25 Oct). About 1,350 words.

## Video presentation (25 Oct 2026)

Limit: 5 min, strict, no speed-up. `video/CM617_video.mp4`: 4:27.9, 1920x1080, 30 fps, 9 slides, about 32 s under the limit.
- Narration: `video/script.py` (spoken form) and `video/narration_script.txt` (readable); 634 words, Windows built-in voice "Microsoft Zira Desktop" at normal rate (`video/make_audio.py`).
- Slides: `video/make_slides.py` -> `video/slides/*.png` (uses the report figures A, B, C).
- Assembly: `video/make_video.py` with ffmpeg from the `imageio-ffmpeg` package (installed into .venv with the user's OK). Each slide stays up for its narration + 0.4 s before + 0.5 s after; the script prints the measured file length.
- Content follows the report: problem, system, 4-step chain, model accuracy, Fig. A, Fig. B, Fig. C with the wallet analogy, conclusion and future scope.
- UPDATE (25 Oct): voice changed to the Microsoft neural voice en-US-GuyNeural via the `edge-tts` package (installed with the user's OK; internet needed; default rate, no speed change). `make_audio.py edge en-US-GuyNeural` (Windows voice kept as `make_audio.py windows Zira`). Added slide 10, a silent credits slide (Akshat Sharma: Simulation; Rudraksh Sharma: Modelling (ML); Suryangshu Das: Inferences; Krishna Gahlod and Sai Charan: Data Collection and Data Processing). Title slide and report author line now list Akshat Sharma, Rudraksh Sharma, Krishna Gahlod, Suryangshu Das, Sai Charan. New length 4:12.98 (10 slides). Fixed garbled bullets and arrows in `make_slides.py` caused by a PowerShell re-save with the wrong encoding.
- FINAL VIDEO (25 Oct): slides made by the user in Google Slides (from `video/gamma_prompt.md` content), exported as PDF, rendered to 1920x1080 PNGs with PyMuPDF (installed in .venv) into `video/slides_gslides/`. All 10 pages checked against the report (numbers, names, figures). Built with `make_video.py slides_gslides CM617_video_final.mp4` using the same Guy narration: `video/CM617_video_final.mp4`, 4:12.98, 1920x1080, 30 fps, sync checked at 2 s, 120 s and the credits start (246.1 s).

### Short answer for the report (OUTDATED: from Task 3, superseded by the main study and the report draft)
With resistance curves from our LFP data, aging cuts the power cap by about 14% at SoH 0.82 (quadratic: 0.856 of fresh). Whether this matters depends on the sizing margin. If the inertia request stays below about 85% of the fresh P_max, aging has no effect on frequency at all. If the battery is sized tight (request near or above the fresh cap), RoCoF gets worse by about 10% at SoH 0.82 while the nadir barely moves. The literature +100% resistance rise would roughly double these effects, but our data does not support it for these cells. Capacity fade only matters if the energy reserve for inertia is very small (about 1 s of rated power).
