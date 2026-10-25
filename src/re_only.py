"""
BEACON main study: a solar-only and a wind-only system where a grid-forming battery is the ONLY source of
inertia and frequency support. Real Kutch 2025 hourly output and site temperature (Renewables.ninja).

System (assumptions): plant 100 MW; battery 25 MW / 50 MWh (nameplate); round-trip efficiency 90 %.
The battery keeps a RESERVE for frequency support (standard for grid-forming batteries):
  power reserve 12.5 MW (design event 10 MW + 25 % margin), charge reserve: SOC never dispatched below 30 %
  (the band 10-30 % is kept for frequency support; 10 % is the absolute floor).
Energy shifting uses the rest: it firms the plant towards a 24-hour centred average of its output.
Load in each hour = plant output + battery dispatch.

Two operating modes:
  SoH-unaware: dispatch limit stays 12.5 MW (nameplate split) as the battery ages, so aging eats the reserve.
  SoH-aware:   dispatch limit = power ceiling(SoH) - 12.5 MW, and the charge kept back is enough to hold
               (dispatch limit + 12.5 MW) for 15 min with the aged capacity (never below 30 %), so both the power
               and the energy reserve are protected and energy shifting shrinks instead.

Stages
  duty   : dispatch of the FRESH battery -> equivalent full cycles per year, depth of discharge, C-rate,
           site temperature during use (inputs to Sandia Model 1)
  aging  : Sandia Model 1 (random forest, no 'days') -> SoH per year for LFP, NMC, NCA
  events : for every year: power ceiling = 25 MW / r(SoH) (Severson quadratic; literature +30 / +100 %
           as sensitivity), energy = 50 MWh x SoH; dispatch rerun with aged limits; in every hour the plant
           produces, a sudden loss of plant output: design 10 % trip (10 MW) and stress 25 % trip (25 MW),
           both capped at what the plant produces.
           covered: the battery's spare power (ceiling - dispatch) covers the loss AND its charge above 10 %
                    covers (dispatch + loss) for 15 minutes
           short:   frequency falls until load damping covers the gap: drop = shortfall / (D x load) x 50 Hz;
                    load shedding if the drop exceeds 0.6 Hz (first stage at 49.4 Hz)
           also: initial RoCoF = loss / (2 x Hv x 25 MW) x 50 Hz (virtual inertia Hv = 4 s on 25 MW); it does
           not depend on SoH as long as the battery covers the loss.
Usage: python re_only.py duty | aging | events | all
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT, FIG = ROOT / "data", ROOT / "outputs", ROOT / "figures"
pd.set_option("display.width", 250)

PLANT_MW = 100.0
BAT_MW, BAT_MWH = 25.0, 50.0
RESERVE_MW = 12.5
SOC_FLOOR, SOC_DISPATCH_MIN, SOC_MAX = 0.10, 0.30, 0.90
ETA = np.sqrt(0.90)                     # one-way efficiency
IST = pd.Timedelta(hours=5, minutes=30)
YEARS = np.arange(0, 16)
CHEMS = ["LFP", "NMC", "NCA"]
CURVES = ["quadratic", "lit_+30%", "lit_+100%"]
FEATS = ["chem_LFP", "chem_NMC", "chem_NCA", "temp_C", "dod", "c_discharge", "efc_nominal"]
EVENTS_MW = {"10% trip (design)": 10.0, "25% trip (stress)": 25.0}
T_HOLD_H = 0.25
MIN_GEN_MW = 1.0
D_LOAD = 1.5                            # load damping, pu/pu (3 %/Hz, as in the grid calibration)
UFLS_HZ = 0.6                           # first load-shedding stage, 49.4 Hz
HV = 4.0
F0 = 50.0


def load_kutch():
    pv = pd.read_csv(DATA / "ninja_pv_23.2500_69.6703_uncorrected.csv", comment="#", parse_dates=["time"])
    wd = pd.read_csv(DATA / "ninja_wind_23.2500_69.6703_uncorrected.csv", comment="#", parse_dates=["time"])
    return pd.DataFrame({"t": pv.time + IST, "solar_cf": pv.electricity.to_numpy(),
                         "wind_cf": wd.electricity.to_numpy(), "temp_C": pv.temperature.to_numpy()})


def aware_soc_min(p_lim_mw, e_mwh, reserve_mw=RESERVE_MW):
    """SoH-aware charge reserve: keep enough charge above the 10 % floor to hold (dispatch limit + power reserve)
    for 15 minutes with the AGED capacity; never below the nameplate 30 %."""
    need = SOC_FLOOR + (p_lim_mw + reserve_mw) * T_HOLD_H / (e_mwh * ETA)
    return min(max(SOC_DISPATCH_MIN, need), SOC_MAX - 0.05)


def dispatch(gen_mw, p_lim_mw, e_mwh, soc0=0.6, soc_min=SOC_DISPATCH_MIN):
    """Hourly firming dispatch within +/- p_lim and SOC soc_min-90 %. Positive p = discharge."""
    target = pd.Series(gen_mw).rolling(24, center=True, min_periods=1).mean().to_numpy()
    n = len(gen_mw)
    p = np.zeros(n); soc = np.zeros(n)
    s = soc0
    for i in range(n):
        soc[i] = s
        want = target[i] - gen_mw[i]
        if want >= 0:
            q = min(want, p_lim_mw, max((s - soc_min) * e_mwh * ETA, 0.0))
            s -= q / ETA / e_mwh
        else:
            q = -min(-want, p_lim_mw, max((SOC_MAX - s) * e_mwh / ETA, 0.0))
            s -= q * ETA / e_mwh
        p[i] = q
    return p, soc


def stage_duty():
    df = load_kutch()
    rows = []
    for src in ["solar", "wind"]:
        gen = PLANT_MW * df[f"{src}_cf"].to_numpy()
        p, soc = dispatch(gen, RESERVE_MW, BAT_MWH)
        dis = np.clip(p, 0, None)
        dod = pd.Series(soc).groupby(df.t.dt.date.to_numpy()).agg(lambda x: x.max() - x.min())
        used = np.abs(p) > 0.5
        rows.append(dict(source=src, efc_per_year=dis.sum() / BAT_MWH, dod_median_pct=100 * float(dod.median()),
                         c_discharge_mean=dis[dis > 0].mean() / BAT_MWH,
                         temp_during_use_C=float(np.average(df.temp_C[used], weights=np.abs(p[used]))),
                         energy_shifted_MWh=float(dis.sum()), plant_MWh=float(gen.sum()),
                         hours_discharging=int((p > 0.5).sum())))
        df[f"{src}_p"], df[f"{src}_soc"] = p, soc
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "re_duty.csv", index=False)
    df.to_csv(OUT / "re_dispatch_fresh.csv", index=False)
    print("Battery duty, fresh battery with reserve (100 MW plant, 25 MW / 50 MWh, dispatch +/-12.5 MW, SOC 30-90 %):")
    print(D.round(3).to_string(index=False))
    for src in ["solar", "wind"]:
        g = df.groupby(df.t.dt.hour)
        print(f"{src}: mean dispatch by hour (MW):", g[f"{src}_p"].mean().round(1).to_dict())
        print(f"{src}: mean SOC by hour:", g[f"{src}_soc"].mean().round(2).to_dict())
    return D


def stage_aging():
    import joblib
    D = pd.read_csv(OUT / "re_duty.csv")
    model = joblib.load(OUT / "models" / "sandia_m1_nodays__forest.joblib")
    pts = pd.read_csv(OUT / "sandia_soh_points.csv")
    pts["efc_nominal"] = pts.cycle_index * pts.dod / 100
    rows = []
    for _, d in D.iterrows():
        for tcase, temp in [("site", d.temp_during_use_C), ("hot 35C", 35.0)]:
            for c in CHEMS:
                max_efc = pts[(pts.chem == c) & (pts.dod >= 60)].efc_nominal.max()
                X = pd.DataFrame({"chem_LFP": int(c == "LFP"), "chem_NMC": int(c == "NMC"), "chem_NCA": int(c == "NCA"),
                                  "temp_C": temp, "dod": d.dod_median_pct, "c_discharge": max(0.5, d.c_discharge_mean),
                                  "efc_nominal": YEARS * d.efc_per_year})
                soh = model.predict(X[FEATS])
                soh[YEARS == 0] = 1.0
                for y, s in zip(YEARS, soh):
                    rows.append(dict(source=d.source, temp_case=tcase, temp_C=temp, chem=c, year=int(y),
                                     efc=y * d.efc_per_year, SoH=float(s), max_tested_efc=max_efc,
                                     beyond_test=bool(y * d.efc_per_year > max_efc)))
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "re_soh.csv", index=False)
    print("Model 1 inputs (solar, wind): DoD", D.dod_median_pct.round(0).tolist(), "%; EFC/yr",
          D.efc_per_year.round(0).tolist(), "; rate", D.c_discharge_mean.round(3).tolist(),
          "-> set to 0.5C (Sandia minimum); temp", D.temp_during_use_C.round(1).tolist())
    print(S[S.temp_case == "site"].pivot_table(index="year", columns=["source", "chem"], values="SoH").round(3).to_string())
    print("first year beyond Sandia test length:",
          S[S.beyond_test & (S.temp_case == "site")].groupby(["source", "chem"]).year.min().to_dict())
    return S


def check_events(gen, p, soc, p_cap, e_mwh):
    load = np.maximum(gen + p, 1e-6)
    res = {}
    for name, mw in EVENTS_MW.items():
        dP = np.minimum(mw, gen)
        active = gen >= MIN_GEN_MW
        headroom = p_cap - p
        power_ok = headroom >= dP
        energy_ok = np.maximum(soc - SOC_FLOOR, 0) * e_mwh * ETA >= np.maximum(p + dP, 0) * T_HOLD_H
        short = np.clip(dP - np.clip(headroom, 0, None), 0, None)
        drop_hz = short / load / D_LOAD * F0
        res[name] = dict(active=active, covered=power_ok & energy_ok, power_ok=power_ok, energy_ok=energy_ok,
                         short=short, drop_hz=drop_hz, shed=(~power_ok) & (drop_hz > UFLS_HZ),
                         rocof0=dP / (2 * HV * BAT_MW) * F0)
    return res


def stage_events():
    import power_sim as ps
    df = load_kutch()
    S = pd.read_csv(OUT / "re_soh.csv")
    hour = df.t.dt.hour.to_numpy()
    rows, prof = [], []
    for (src, tcase, chem, year), s in S.groupby(["source", "temp_case", "chem", "year"]):
        soh = float(s.SoH.iloc[0])
        gen = PLANT_MW * df[f"{src}_cf"].to_numpy()
        for curve in CURVES:
            if year == 0 and curve != "quadratic":
                continue
            r = float(ps.r_curve(curve, soh))
            p_cap, e = BAT_MW / r, BAT_MWH * soh
            for mode in ["SoH-unaware", "SoH-aware"]:
                p_lim = RESERVE_MW if mode == "SoH-unaware" else max(p_cap - RESERVE_MW, 0.0)
                p_lim = min(p_lim, p_cap)
                soc_min = SOC_DISPATCH_MIN if mode == "SoH-unaware" else aware_soc_min(p_lim, e)
                p, soc = dispatch(gen, p_lim, e, soc_min=soc_min)
                ev = check_events(gen, p, soc, p_cap, e)
                for name, x in ev.items():
                    a = x["active"]
                    rows.append(dict(source=src, temp_case=tcase, chem=chem, year=int(year), curve=curve, operation=mode,
                                     event=name, SoH=soh, r=r, p_cap_MW=p_cap, e_MWh=e, dispatch_limit_MW=p_lim,
                                     beyond_test=bool(s.beyond_test.iloc[0]), event_hours=int(a.sum()),
                                     pct_not_covered=100 * (~x["covered"][a]).mean(),
                                     pct_power_short=100 * (~x["power_ok"][a]).mean(),
                                     pct_energy_short=100 * (~x["energy_ok"][a]).mean(),
                                     pct_load_shedding=100 * x["shed"][a].mean(),
                                     max_drop_hz=float(x["drop_hz"][a].max()),
                                     energy_shifted_MWh=float(np.clip(p, 0, None).sum()),
                                     rocof0_median=float(np.median(x["rocof0"][a]))))
                    if curve == "quadratic" and tcase == "site" and year in (0, 5, 10):
                        for h in range(24):
                            m = a & (hour == h)
                            prof.append(dict(source=src, chem=chem, year=int(year), operation=mode, event=name, hour=h,
                                             event_hours=int(m.sum()),
                                             pct_not_covered=100 * (~x["covered"][m]).sum() / max(m.sum(), 1)))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "re_events.csv", index=False)
    pd.DataFrame(prof).to_csv(OUT / "re_events_by_hour.csv", index=False)
    print("event hours per year (plant producing >= 1 MW):", R[R.year == 0].groupby("source").event_hours.first().to_dict())
    base = (R.temp_case == "site") & (R.curve == "quadratic")
    for ev in EVENTS_MW:
        for mode in ["SoH-unaware", "SoH-aware"]:
            x = R[base & (R.event == ev) & (R.operation == mode)]
            print(f"\n===== {ev}, {mode}: % of event hours NOT covered (and % with load shedding) =====")
            t = x.pivot_table(index="year", columns=["source", "chem"], values="pct_not_covered").round(2)
            print(t.to_string())
            print("load shedding %:", x[x.year.isin([0, 5, 10, 15])].pivot_table(
                index="year", columns=["source", "chem"], values="pct_load_shedding").round(2).to_string())
    print("\nenergy shifted per year (MWh), quadratic, site, design event rows:")
    e = R[base & (R.event == "10% trip (design)") & R.year.isin([0, 5, 10, 15])]
    print(e.pivot_table(index="year", columns=["source", "chem", "operation"], values="energy_shifted_MWh").round(0).to_string())
    print("\nyear 10, design event, SoH-unaware, % not covered by R(SoH) curve:")
    print(R[(R.temp_case == "site") & (R.year == 10) & (R.event == "10% trip (design)") & (R.operation == "SoH-unaware")]
          .pivot_table(index="curve", columns=["source", "chem"], values="pct_not_covered").round(2).to_string())
    print("\nyear 10, design event, SoH-unaware, quadratic: site vs hot 35C:")
    print(R[(R.curve == "quadratic") & (R.year == 10) & (R.event == "10% trip (design)") & (R.operation == "SoH-unaware")]
          .pivot_table(index="temp_case", columns=["source", "chem"], values="pct_not_covered").round(2).to_string())
    print("\ninitial RoCoF when covered (Hz/s, median):", R[R.year == 0].groupby(["source", "event"]).rocof0_median.first().round(2).to_dict())
    return R


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("duty", "all"):
        stage_duty()
    if stage in ("aging", "all"):
        stage_aging()
    if stage in ("events", "all"):
        stage_events()

