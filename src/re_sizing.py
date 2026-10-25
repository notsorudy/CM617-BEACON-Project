"""
BEACON: battery and reserve sizing sweep for the solar-only / wind-only study (same method as re_only.py).
Plant 100 MW. Battery sizes kept realistic for a plant of this size:
  power 15 / 25 / 35 MW (15-35 % of plant), duration 1 / 2 / 4 h (energy = power x duration),
  power reserve 10 / 12.5 / 15 MW (design trip 10 MW + 0 / 25 / 50 % margin); combinations with reserve >= power skipped.
For every size: duty of the fresh battery -> Sandia Model 1 SoH per year -> quadratic R(SoH) power ceiling ->
dispatch rerun with aged limits (SoH-unaware and SoH-aware) -> design (10 MW) and stress (25 MW) events.
Site temperature and the quadratic R(SoH) curve only.
Usage: python re_sizing.py
"""
import joblib
import numpy as np
import pandas as pd

import power_sim as ps
from re_only import (OUT, PLANT_MW, SOC_FLOOR, SOC_DISPATCH_MIN, ETA, YEARS, CHEMS, FEATS, EVENTS_MW,
                     load_kutch, dispatch, check_events, aware_soc_min)

POWERS = [15.0, 25.0, 35.0]
HOURS = [1.0, 2.0, 4.0]
RESERVES = [10.0, 12.5, 15.0]


def duty(df, gen, p_lim, e):
    p, soc = dispatch(gen, p_lim, e)
    dis = np.clip(p, 0, None)
    dod = pd.Series(soc).groupby(df.t.dt.date.to_numpy()).agg(lambda x: x.max() - x.min())
    used = np.abs(p) > 0.5
    temp = float(np.average(df.temp_C[used], weights=np.abs(p[used]))) if used.any() else float(df.temp_C.mean())
    return dict(efc_per_year=dis.sum() / e, dod_median_pct=100 * float(dod.median()),
                c_discharge_mean=float(dis[dis > 0].mean() / e) if (dis > 0).any() else 0.0, temp_C=temp)


def main():
    df = load_kutch()
    model = joblib.load(OUT / "models" / "sandia_m1_nodays__forest.joblib")
    pts = pd.read_csv(OUT / "sandia_soh_points.csv")
    pts["efc_nominal"] = pts.cycle_index * pts.dod / 100
    rows = []
    for bat_mw in POWERS:
        for hrs in HOURS:
            e0 = bat_mw * hrs
            for res_mw in RESERVES:
                if res_mw >= bat_mw:
                    continue
                for src in ["solar", "wind"]:
                    gen = PLANT_MW * df[f"{src}_cf"].to_numpy()
                    d = duty(df, gen, bat_mw - res_mw, e0)
                    for c in CHEMS:
                        max_efc = pts[(pts.chem == c) & (pts.dod >= min(d["dod_median_pct"], 60))].efc_nominal.max()
                        X = pd.DataFrame({"chem_LFP": int(c == "LFP"), "chem_NMC": int(c == "NMC"),
                                          "chem_NCA": int(c == "NCA"), "temp_C": d["temp_C"], "dod": d["dod_median_pct"],
                                          "c_discharge": max(0.5, d["c_discharge_mean"]),
                                          "efc_nominal": YEARS * d["efc_per_year"]})
                        soh = model.predict(X[FEATS])
                        soh[YEARS == 0] = 1.0
                        for y, s in zip(YEARS, soh):
                            r = float(ps.r_curve("quadratic", s))
                            p_cap, e = bat_mw / r, e0 * s
                            for mode in ["SoH-unaware", "SoH-aware"]:
                                p_lim = bat_mw - res_mw if mode == "SoH-unaware" else max(p_cap - res_mw, 0.0)
                                p_lim = min(p_lim, p_cap)
                                soc_min = (SOC_DISPATCH_MIN if mode == "SoH-unaware"
                                           else aware_soc_min(p_lim, e, res_mw))
                                p, soc = dispatch(gen, p_lim, e, soc_min=soc_min)
                                ev = check_events(gen, p, soc, p_cap, e)
                                for name, x in ev.items():
                                    a = x["active"]
                                    rows.append(dict(bat_MW=bat_mw, hours=hrs, bat_MWh=e0, reserve_MW=res_mw, source=src,
                                                     chem=c, year=int(y), operation=mode, event=name,
                                                     efc_per_year=d["efc_per_year"], dod_pct=d["dod_median_pct"],
                                                     c_rate=d["c_discharge_mean"], temp_C=d["temp_C"], SoH=float(s),
                                                     beyond_test=bool(y * d["efc_per_year"] > max_efc), p_cap_MW=p_cap,
                                                     pct_not_covered=100 * (~x["covered"][a]).mean(),
                                                     pct_power_short=100 * (~x["power_ok"][a]).mean(),
                                                     pct_energy_short=100 * (~x["energy_ok"][a]).mean(),
                                                     pct_load_shedding=100 * x["shed"][a].mean(),
                                                     energy_shifted_MWh=float(np.clip(p, 0, None).sum())))
                print(f"done {bat_mw:.0f} MW x {hrs:.0f} h", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "re_sizing.csv", index=False)

    key = ["bat_MW", "hours", "reserve_MW"]
    d0 = R[(R.year == 0) & (R.chem == "LFP") & (R.operation == "SoH-unaware") & (R.event == "10% trip (design)")]
    print("\n===== duty of the fresh battery (cycles/yr, DoD %, C-rate) =====")
    print(d0.pivot_table(index=key, columns="source", values=["efc_per_year", "dod_pct", "c_rate"]).round(2).to_string())
    y10 = R[(R.year == 10) & (R.operation == "SoH-unaware") & (R.event == "10% trip (design)")]
    print("\n===== SoH at year 10 =====")
    print(y10.pivot_table(index=key, columns=["source", "chem"], values="SoH").round(3).to_string())
    for ev in EVENTS_MW:
        for mode in ["SoH-unaware", "SoH-aware"]:
            x = R[(R.year == 10) & (R.operation == mode) & (R.event == ev)]
            print(f"\n===== year 10, {ev}, {mode}: % not covered | % load shedding =====")
            print(x.pivot_table(index=key, columns=["source", "chem"], values="pct_not_covered").round(2).to_string())
            print(x.pivot_table(index=key, columns=["source", "chem"], values="pct_load_shedding").round(2).to_string())
    x = R[(R.year == 10) & (R.event == "10% trip (design)")]
    t = x.pivot_table(index=key + ["source", "chem"], columns="operation", values="energy_shifted_MWh")
    t["cost_pct"] = 100 * (1 - t["SoH-aware"] / t["SoH-unaware"])
    print("\n===== year 10: energy shifted, SoH-aware cost (%) =====")
    print(t.cost_pct.unstack(["source", "chem"]).round(2).to_string())
    return R


if __name__ == "__main__":
    main()
