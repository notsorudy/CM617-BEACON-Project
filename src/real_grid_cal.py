"""
BEACON: rerun of the real-grid hours and of the per-chemistry chain with the CALIBRATED grid model
(grid_cal.py, fitted to the 21 Jan 2019 Grid-India event and POSOCO's measured response).

Changes from real_grid_sim.py / chain_sim.py:
  - H = h_s x sync share, governor K = k_s x sync share, two-stage governor (0.8 s) and turbine (10 s),
    load damping D, all from outputs/grid_cal_params.json
  - events are real Indian sizes: 4,500 MW (IEGC reference contingency) and 5,346 MW (largest observed,
    28 May 2020), as a share of each hour's real demand
  - battery sized against the hour with the largest peak inertia request for the 4,500 MW event:
    peak request = Hv / (H + Hv) x dP; fresh P_max = peak request / d, d in {0.8, 1.0, 1.2}
Unchanged: R(SoH) curves (power_sim.r_curve), battery logic, energy reserve 10 s x SoH, Hv = 2 s,
  Sandia Model 1 SoH table (outputs/chain_soh.csv).
Speed: hours are grouped by (synchronous share rounded to 0.01, event size rounded to 0.0005 pu).

Part A: all clean hours 2021-2025, SoH 1.0 / 0.9 / 0.82 / 0.7, curves quadratic / linear / lit +100 %.
Part B: chain per chemistry on the latest 12 months (Jul 2024 - Jun 2025), 4,500 MW event.
Outputs: outputs/real_grid_cal_*.csv, outputs/chain_cal_results.csv, figures/real_grid_cal_*.png,
figures/chain_cal_hours_years.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import grid_cal as gc
import power_sim as ps

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "outputs", ROOT / "figures"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
pd.set_option("display.width", 250)

P = json.load(open(OUT / "grid_cal_params.json"))
EVENTS = {"4500MW": 4500.0, "5346MW": 5346.0}
SOHS = [1.0, 0.9, 0.82, 0.7]
CURVES = ["quadratic", "linear", "lit_+100%"]
DEMANDS = [0.8, 1.0, 1.2]
T_RES = 10.0
SYNC_BIN, DP_BIN = 0.01, 0.0005
CHUNK = 40000


def load_hours():
    g = pd.read_csv(OUT / "grid_india_hourly_clean.csv", parse_dates=["time"])
    g = g[g.valid].copy()
    g["sync"] = g.sync_share.clip(upper=1.0)
    g["sb"] = (np.round(g.sync / SYNC_BIN) * SYNC_BIN).round(4)
    g["year"] = g.time.dt.year
    for k, mw in EVENTS.items():
        g[f"dp_{k}"] = (np.round(mw / g.demand_MW / DP_BIN) * DP_BIN).round(5)
    return g


def run_cases(C, t_end=60.0):
    """C: DataFrame with sb, dp, event, p_mw (fresh battery P_max in MW; 0 = no battery), r, soh.
    The battery has a fixed size in MW, so its per-unit cap differs by hour: since dp = event_MW / demand,
    cap (pu of that hour's demand) = p_mw / demand = p_mw x dp / event_MW."""
    C = C.copy()
    C["p_cap0"] = C.p_mw * C.dp / C.event.map(EVENTS)
    outs = []
    for i in range(0, len(C), CHUNK):
        c = C.iloc[i:i + CHUNK]
        for bat_on, cc in [(True, c[c.p_mw > 0]), (False, c[c.p_mw <= 0])]:
            if cc.empty:
                continue
            h = P["h_s"] * cc.sb.to_numpy()
            k = P["k_s"] * cc.sb.to_numpy()
            d = np.full(len(cc), P["d"])
            if bat_on:
                res, _ = gc.simulate(cc.dp.to_numpy(), h, k, d, p_cap=cc.p_cap0.to_numpy(), r=cc.r.to_numpy(),
                                     e_avail=T_RES * cc.p_cap0.to_numpy() * cc.soh.to_numpy(), battery=True,
                                     t_end=t_end)
            else:   # no battery at all
                res, _ = gc.simulate(cc.dp.to_numpy(), h, k, d, battery=False, t_end=t_end)
                res.update(saturated=np.zeros(len(cc), bool), sat_time_s=np.zeros(len(cc)),
                           energy_limit_hit=np.zeros(len(cc), bool), p_bat_peak=np.zeros(len(cc)),
                           energy_used_peak=np.zeros(len(cc)))
            outs.append(pd.DataFrame({k2: v for k2, v in res.items()}, index=cc.index))
        print(f"   simulated {min(i + CHUNK, len(C))}/{len(C)}", flush=True)
    return C.join(pd.concat(outs))


def sizing(g):
    """Peak inertia request in MW for the 4,500 MW event, largest over all clean hours."""
    H = P["h_s"] * g.sb
    req_mw = P["Hv"] / (H + P["Hv"]) * 4500.0
    i = req_mw.idxmax()
    return float(req_mw.max()), g.loc[i]


def part_a(g, peak_req):
    pairs = pd.concat([g[["sb", f"dp_{k}"]].rename(columns={f"dp_{k}": "dp"}).assign(event=k) for k in EVENTS])
    pairs = pairs.drop_duplicates().reset_index(drop=True)
    rows = []
    for d in DEMANDS:
        for c in CURVES:
            for s in SOHS:
                if s == 1.0 and c != "quadratic":
                    continue
                rows.append(dict(design_demand=d, curve=c, soh=s, r=float(ps.r_curve(c, s)), p_mw=peak_req / d))
    B = pd.DataFrame(rows)
    C = pairs.merge(B, how="cross")
    nb = pairs.assign(design_demand=np.nan, curve="no battery", soh=1.0, r=1.0, p_mw=0.0)
    print(f"Part A: {len(pairs)} grid states x {len(B)} battery cases + no-battery = {len(C) + len(nb)} runs")
    C = run_cases(pd.concat([C, nb], ignore_index=True))
    C.to_csv(OUT / "real_grid_cal_cases.csv", index=False)

    summ = []
    hourly = g[["time", "year", "sync", "sb", "demand_MW", "vre_share"] + [f"dp_{k}" for k in EVENTS]].copy()
    for ev in EVENTS:
        key = g[["sb", f"dp_{ev}"]].rename(columns={f"dp_{ev}": "dp"})
        nbt = C[(C.curve == "no battery") & (C.event == ev)].set_index(["sb", "dp"])
        x = nbt.reindex(pd.MultiIndex.from_frame(key))
        hourly[f"nadir_{ev}_nobattery"] = 50 + x.nadir_dev_Hz.to_numpy()
        hourly[f"rocof_{ev}_nobattery"] = x.rocof_500ms_Hz_s.to_numpy()
        for d in DEMANDS:
            fr = C[(C.event == ev) & (C.design_demand == d) & (C.soh == 1.0)].set_index(["sb", "dp"]) \
                .reindex(pd.MultiIndex.from_frame(key))
            for c in CURVES:
                for s in SOHS[1:]:
                    a = C[(C.event == ev) & (C.design_demand == d) & (C.curve == c) & (C.soh == s)] \
                        .set_index(["sb", "dp"]).reindex(pd.MultiIndex.from_frame(key))
                    dro = a.rocof_500ms_Hz_s.to_numpy() - fr.rocof_500ms_Hz_s.to_numpy()
                    dna = a.nadir_dev_Hz.to_numpy() - fr.nadir_dev_Hz.to_numpy()
                    matters = (np.abs(dro) > 1e-3) | (np.abs(dna) > 1e-3)
                    for yr, m in [(y, (g.year == y).to_numpy()) for y in sorted(g.year.unique())] + [("all", np.ones(len(g), bool))]:
                        summ.append(dict(event=ev, design_demand=d, curve=c, soh=s, year=yr,
                                         pct_hours_aging_matters=100 * matters[m].mean(),
                                         pct_hours_saturated=100 * a.saturated.to_numpy()[m].mean(),
                                         pct_energy_hit=100 * a.energy_limit_hit.to_numpy()[m].mean(),
                                         drocof_max=np.nanmax(dro[m]), dnadir_min=np.nanmin(dna[m]),
                                         nadir_min_aged=50 + np.nanmin(a.nadir_dev_Hz.to_numpy()[m]),
                                         nadir_min_fresh=50 + np.nanmin(fr.nadir_dev_Hz.to_numpy()[m]),
                                         rocof_max_aged=np.nanmax(a.rocof_500ms_Hz_s.to_numpy()[m]),
                                         rocof_max_fresh=np.nanmax(fr.rocof_500ms_Hz_s.to_numpy()[m])))
                    if c == "quadratic" and d == 1.0:
                        hourly[f"nadir_{ev}_soh{s}"] = 50 + a.nadir_dev_Hz.to_numpy()
                        hourly[f"rocof_{ev}_soh{s}"] = a.rocof_500ms_Hz_s.to_numpy()
                        hourly[f"sat_{ev}_soh{s}"] = a.saturated.to_numpy()
            if d == 1.0:
                hourly[f"nadir_{ev}_soh1.0"] = 50 + fr.nadir_dev_Hz.to_numpy()
                hourly[f"rocof_{ev}_soh1.0"] = fr.rocof_500ms_Hz_s.to_numpy()
    S = pd.DataFrame(summ)
    S.to_csv(OUT / "real_grid_cal_summary.csv", index=False)
    hourly.to_csv(OUT / "real_grid_cal_hourly.csv", index=False)
    return C, S, hourly


def part_b(g, peak_req):
    soh = pd.read_csv(OUT / "chain_soh.csv")
    w = g[g.time >= "2024-07-01"]
    pairs = w[["sb", "dp_4500MW"]].rename(columns={"dp_4500MW": "dp"}).drop_duplicates().reset_index(drop=True)
    pairs["event"] = "4500MW"
    curves = ["quadratic", "lit_+30%", "lit_+100%"]
    soh = soh.assign(key=1).merge(pd.DataFrame({"curve": curves, "key": 1}), on="key").drop(columns="key")
    soh["r"] = [float(ps.r_curve(c, s)) for c, s in zip(soh.curve, soh.SoH)]
    soh["r_key"] = soh.r.round(4)
    rk = np.unique(np.append(soh.r_key.to_numpy(), 1.0))
    B = pd.DataFrame([dict(design_demand=d, r=r, r_key=r, soh=1.0, p_mw=peak_req / d) for d in DEMANDS for r in rk])
    C = pairs.merge(B, how="cross")
    print(f"Part B: {len(pairs)} grid states x {len(B)} battery cases = {len(C)} runs")
    C = run_cases(C, t_end=40.0)
    key = w[["sb", "dp_4500MW"]].rename(columns={"dp_4500MW": "dp"})
    hod = w.time.dt.hour.to_numpy()
    rows = []
    for d in DEMANDS:
        Cd = C[C.design_demand == d]
        fr = Cd[Cd.r_key == 1.0].set_index(["sb", "dp"]).reindex(pd.MultiIndex.from_frame(key))
        for r_, Cr in Cd.groupby("r_key"):
            a = Cr.set_index(["sb", "dp"]).reindex(pd.MultiIndex.from_frame(key))
            dro = a.rocof_500ms_Hz_s.to_numpy() - fr.rocof_500ms_Hz_s.to_numpy()
            dna = a.nadir_dev_Hz.to_numpy() - fr.nadir_dev_Hz.to_numpy()
            matters = (np.abs(dro) > 1e-3) | (np.abs(dna) > 1e-3)
            rows.append(dict(design_demand=d, r_key=r_, pct_hours_aging_matters=100 * matters.mean(),
                             pct_hours_saturated=100 * a.saturated.to_numpy().mean(),
                             drocof_max=np.nanmax(dro), dnadir_min=np.nanmin(dna),
                             nadir_min=50 + np.nanmin(a.nadir_dev_Hz.to_numpy())))
    R = pd.DataFrame(rows)
    full = soh.merge(R, on="r_key")
    full.to_csv(OUT / "chain_cal_results.csv", index=False)
    return full


def figures(g, S, hourly, full, size_hour):
    yrs = sorted(g.year.unique())
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for a, d in zip(ax, DEMANDS):
        for c, ls in [("quadratic", "-"), ("linear", "-."), ("lit_+100%", "--")]:
            for s, col in [(0.9, "tab:blue"), (0.82, "tab:orange"), (0.7, "tab:red")]:
                x = S[(S.event == "4500MW") & (S.design_demand == d) & (S.curve == c) & (S.soh == s) & (S.year != "all")]
                x = x.set_index("year").reindex(yrs)
                a.plot([str(y) for y in yrs], x.pct_hours_aging_matters, ls=ls, color=col, marker="o", ms=3,
                       label=f"{c}, SoH {s}" + (" (extrap.)" if s < ps.SOH_DATA_MIN else ""))
        a.set(title=f"battery sized at demand ratio {d}", xlabel="year (2021: Sep-Dec, 2025: Jan-Jun)")
    ax[0].set_ylabel("% of hours where aging matters")
    ax[0].legend(fontsize=7)
    fig.suptitle("CALIBRATED grid, 4,500 MW loss, real Grid-India hours")
    fig.tight_layout()
    fig.savefig(FIG / "real_grid_cal_aging_hours_by_year.png", dpi=150)
    plt.close(fig)

    y = hourly[hourly.year == 2025]
    hod = y.time.dt.hour
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    for col, lab, cl in [("nobattery", "no battery", "grey"), ("soh1.0", "SoH 1.0", "k"), ("soh0.9", "SoH 0.9", "tab:blue"),
                         ("soh0.82", "SoH 0.82", "tab:orange")]:
        ax[0].plot(y.groupby(hod)[f"nadir_4500MW_{col}"].mean(), color=cl, label=lab)
        ax[1].plot(y.groupby(hod)[f"rocof_4500MW_{col}"].mean(), color=cl, label=lab)
    ax[0].set(xlabel="hour of day (IST)", ylabel="mean nadir (Hz)", title="2025, 4,500 MW loss, quadratic, d = 1.0")
    ax[1].set(xlabel="hour of day (IST)", ylabel="mean RoCoF 500 ms (Hz/s)", title="same, RoCoF")
    for a in ax:
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "real_grid_cal_2025_by_hour.png", dpi=150)
    plt.close(fig)

    # time traces in the sizing hour: no battery, fresh, aged
    s, dp = float(size_hour.sb), float(size_hour["dp_4500MW"])
    peak = P["Hv"] / (P["h_s"] * s + P["Hv"]) * 4500.0 / float(size_hour.demand_MW)   # pu of that hour
    rs = [1.0, float(ps.r_curve("quadratic", 0.9)), float(ps.r_curve("quadratic", 0.82)), float(ps.r_curve("lit_+100%", 0.82))]
    labs = ["fresh (SoH 1.0)", "SoH 0.9", "SoH 0.82", "SoH 0.82, lit +100 %"]
    sohs = [1.0, 0.9, 0.82, 0.82]
    n = len(rs)
    res, tr = gc.simulate(np.full(n, dp), np.full(n, P["h_s"] * s), np.full(n, P["k_s"] * s), np.full(n, P["d"]),
                          p_cap=np.full(n, peak), r=np.array(rs), e_avail=T_RES * peak * np.array(sohs), battery=True,
                          keep=range(n))
    _, trn = gc.simulate(np.array([dp]), np.array([P["h_s"] * s]), np.array([P["k_s"] * s]), np.array([P["d"]]), keep=[0])
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    t0 = np.array(trn[0])
    ax[0].plot(t0[:, 0], 50 + t0[:, 1], color="grey", label="no battery")
    for i in range(n):
        t = np.array(tr[i])
        ax[0].plot(t[:, 0], 50 + t[:, 1], label=labs[i])
        ax[1].plot(t[:, 0], t[:, 2] * size_hour.demand_MW, label=labs[i])
    ax[0].set(xlim=(0, 40), xlabel="time (s)", ylabel="frequency (Hz)",
              title=f"hardest hour {size_hour.time:%d %b %Y %H:%M}, 4,500 MW lost, d = 1.0")
    ax[1].set(xlim=(0, 40), xlabel="time (s)", ylabel="battery power (MW)", title="battery power (capped at P_max(SoH))")
    for a in ax:
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "real_grid_cal_trace_hardest_hour.png", dpi=150)
    plt.close(fig)

    COL = {"LFP": "tab:blue", "NMC": "tab:orange", "NCA": "tab:green"}
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)
    for a, sc in zip(ax, full.scenario.unique()):
        for c in ["LFP", "NMC", "NCA"]:
            for curve, ls in [("quadratic", "-"), ("lit_+100%", "--")]:
                x = full[(full.scenario == sc) & (full.chem == c) & (full.curve == curve) & (full.design_demand == 1.0)] \
                    .sort_values("year")
                a.plot(x.year, x.pct_hours_aging_matters, ls=ls, color=COL[c], marker="o", ms=3, label=f"{c}, {curve}")
                b = x[x.beyond_test]
                a.plot(b.year, b.pct_hours_aging_matters, "o", color="white", mec=COL[c], ms=6)
        a.set(title=sc, xlabel="years at 1 equivalent full cycle per day")
    ax[0].set_ylabel("% of hours where aging matters")
    ax[0].legend(fontsize=7, title="open marker: beyond Sandia test length", title_fontsize=7)
    fig.suptitle("CALIBRATED grid, chain per chemistry, 4,500 MW loss, d = 1.0, Jul 2024 - Jun 2025 hours")
    fig.tight_layout()
    fig.savefig(FIG / "chain_cal_hours_years.png", dpi=150)
    plt.close(fig)


def main():
    g = load_hours()
    peak_req, hour = sizing(g)
    print(f"clean hours {len(g)}; hardest hour (lowest inertia) {hour.time} (sync {hour.sync:.3f}, demand "
          f"{hour.demand_MW:.0f} MW, H {P['h_s'] * hour.sb:.2f} s); peak inertia request for 4,500 MW = "
          f"{peak_req:.0f} MW; fresh battery P_max = {peak_req:.0f} MW / d")
    print("H by year (s), median / min:", (P["h_s"] * g.groupby("year").sync.median()).round(2).to_dict(),
          (P["h_s"] * g.groupby("year").sync.min()).round(2).to_dict())
    C, S, hourly = part_a(g, peak_req)
    for ev in EVENTS:
        nb = hourly[f"nadir_{ev}_nobattery"]
        print(f"\nno battery, {ev}: nadir {nb.min():.3f} to {nb.max():.3f} Hz; RoCoF {hourly[f'rocof_{ev}_nobattery'].min():.3f} "
              f"to {hourly[f'rocof_{ev}_nobattery'].max():.3f} Hz/s")
        fr = hourly[f"nadir_{ev}_soh1.0"]
        print(f"fresh battery d=1.0, {ev}: nadir {fr.min():.3f} to {fr.max():.3f} Hz; RoCoF "
              f"{hourly[f'rocof_{ev}_soh1.0'].min():.3f} to {hourly[f'rocof_{ev}_soh1.0'].max():.3f}")
    show = ["design_demand", "curve", "soh", "pct_hours_aging_matters", "pct_hours_saturated", "pct_energy_hit",
            "drocof_max", "dnadir_min", "nadir_min_fresh", "nadir_min_aged"]
    for ev in EVENTS:
        print(f"\n===== {ev}, all hours =====")
        print(S[(S.event == ev) & (S.year == "all")][show].round(4).to_string(index=False))
    print("\n===== % hours aging matters by year, quadratic, 4,500 MW =====")
    q = S[(S.event == "4500MW") & (S.curve == "quadratic") & (S.year != "all")]
    print(q.pivot_table(index=["design_demand", "soh"], columns="year", values="pct_hours_aging_matters").round(2).to_string())
    import sys
    if "--part-a-only" in sys.argv:   # Part B has no no-battery rows; reuse its saved results
        full = pd.read_csv(OUT / "chain_cal_results.csv")
    else:
        full = part_b(g, peak_req)
    qq = full[(full.curve == "quadratic") & (full.design_demand == 1.0)]
    for sc in qq.scenario.unique():
        print(f"\n=== chain, {sc}, quadratic, d = 1.0 ===")
        print(qq[qq.scenario == sc].pivot_table(index="year", columns="chem",
                                                values=["SoH", "pct_hours_aging_matters", "dnadir_min", "drocof_max"])
              .round(3).to_string())
    figures(g, S, hourly, full, hour)
    print("\nsaved outputs and figures")


if __name__ == "__main__":
    main()
