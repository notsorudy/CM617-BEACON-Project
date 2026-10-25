# BEACON: A State-of-Health-Aware Framework for Synthetic Inertia Delivery in Solar and Wind-Integrated Grids

CM617 course project, IIT Bombay.
**Authors:** Akshat Sharma, Rudraksh Sharma, Krishna Gahlod, Suryangshu Das, Sai Charan

## What this project does

Grid-forming batteries can supply synthetic inertia where solar and wind replace spinning generators, but most studies assume the battery always delivers its rated power. We model how battery ageing limits that power, in a system powered only by solar or only by wind, where the battery is the only source of inertia.

The chain:
1. **Real data:** hourly 2025 solar and wind output and temperature for Kutch, Gujarat (Renewables.ninja) set how hard the battery works.
2. **Model 1 (state of health):** a random forest trained on the Sandia cycling data (85 cells; LFP, NMC, NCA) predicts SoH per year from chemistry, temperature, depth of discharge, discharge rate and cycles.
3. **Model 2 (resistance and power):** a quadratic R(SoH) curve fitted on 124 Severson LFP cells gives resistance, and so the maximum power (P_max ∝ 1/R).
4. **Events:** in every producing hour a 10 MW trip is applied to a 100 MW plant with a 25 MW / 50 MWh grid-forming battery, and the first 20 s of frequency are simulated.

## Key results (year 10)

| | LFP | NMC | NCA |
|---|---|---|---|
| SoH | 0.95 | 0.84 | 0.82 |
| Power ceiling (of 25 MW) | 24.3 MW | 21.8 MW | 21.5 MW |
| Trips not covered, solar / wind | 0% / 0% | 0.45% / 1.14% | 0.59% / 1.24% |
| Trips not covered, SoH-aware operation | 0% / 0% | 0% / 0% | 0% / 0% |

- Within its power limit, an aged battery gives the same frequency response as a new one (dip to 49.81 Hz, recovery). Past the limit, frequency crosses 49.4 Hz in under 1 s and does not recover.
- Wind is about twice as exposed as solar (twice as many producing hours).
- SoH-aware operation (protecting the reserve as the battery ages) removes every failure at almost no cost.
- Model accuracy (R² on unseen data): Model 1, 0.87 (unseen cells) and 0.81 (unseen test conditions); Model 2, quadratic 0.78 vs linear formula 0.74.

Full log of every step, number and assumption: [PROGRESS.md](PROGRESS.md). Report source: [report/main.tex](report/main.tex).

## Repository layout

```
src/        all analysis scripts (data, models, simulations, figures)
outputs/    result tables, run logs, and the two trained models the main chain uses
figures/    all figures
report/     2-page report (LaTeX) and its figures
video/      narration script, slide prompt, and scripts that build the presentation video
data/       only the small Kutch Renewables.ninja files and calibration targets (raw data: see below)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data (not included: about 16.8 GB)

| Data | Where to get it | Put it in |
|---|---|---|
| Severson et al. (2019) LFP cycling data, 3 batch `.mat` files | https://data.matr.io/1/ | `data/` |
| Sandia (Preger et al. 2020) cycling data | batteryarchive.org; `src/download_sandia.py` downloads it (set `BATTERYARCHIVE_KEY` to the public key from the dashboard's share link) | `data/sandia/` |
| Renewables.ninja PV and wind, Kutch (23.25 N, 69.67 E), 2025 | included | `data/` |
| Grid-India demand and generation, POSOCO 2022 inertia report (national-grid context study only) | Grid-India / POSOCO websites | `data/demand/`, `data/grid_events/` |

## How to reproduce

Run from `src/`, roughly in this order:

1. **Severson data and Model 2:** `build_pkl.py`, `preprocess_severson.py`, `models.py`, `linearity.py` (writes `outputs/resistance_curve_coefs.json`), `power_sim.py`
2. **Sandia data and Model 1:** `download_sandia.py`, `sandia_soh.py`, `sandia_model1.py`, `sandia_model1_nodays.py` (main model)
3. **Main study (solar-only / wind-only):** `re_only.py all`, `plot_re_only.py`
4. **Battery and reserve sizing sweep:** `re_sizing.py`, `plot_re_sizing.py`
5. **Frequency response over time:** `re_freq.py`, `plot_re_freq.py`
6. **Report figures:** `plot_report.py`
7. **Video:** in `video/`: `make_audio.py`, `make_slides.py`, `make_video.py`

The national-grid context study (`build_grid_hourly.py`, `clean_grid_data.py`, `grid_cal.py`, `real_grid_cal.py`, `chain_sim.py`) is kept for reference; it is not part of the final results.

## Limitations

- The resistance curve comes from LFP cells and is used for all three chemistries.
- Lab 18650 cells, not grid-scale cells; Sandia tests end before NMC and NCA reach the later years of this duty.
- Load is set equal to plant plus battery output (no measured demand exists for such systems).
- The frequency model uses assumed control settings and has no recorded event to calibrate against.

## Data credits

Severson et al., *Nature Energy* 4, 383-391 (2019); Preger et al., *J. Electrochem. Soc.* 167, 120532 (2020), via batteryarchive.org; Renewables.ninja (Pfenninger and Staffell 2016; Staffell and Pfenninger 2016), CC BY-NC 4.0.
