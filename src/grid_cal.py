"""
BEACON: grid frequency model calibrated to the Indian grid (POSOCO / Grid-India 2022 inertia report).

Model (per unit on the demand base of the hour, f0 = 50 Hz):
  2 H dw/dt   = Pm - dP - D w + P_bat
  Tg dxg/dt   = -K w - xg          (governor, first order, Tg = 0.8 s as in POSOCO's model)
  Tt dPm/dt   = xg - Pm            (turbine, first order, Tt = 10 s as in POSOCO's model)
  H = h_s x sync_share, K = k_s x sync_share (only synchronous plants give inertia and governor response)
  P_bat = -2 Hv dw/dt, clipped to +/- P_cap and to the energy left (same battery logic as power_sim.py)

Calibration targets (data/grid_events/calibration_targets.csv):
  event A, 21 Jan 2019 15:57: 780 MW lost of 144,482 MW; initial RoCoF 0.0167 Hz/s; nadir drop 0.081 Hz
  after 10.6 s.  Steady-state frequency response about 15,000 MW/Hz (median, POSOCO). Load damping 2-5 %/Hz.
Checks: time to nadir 9-14 s; 4,500 MW reference contingency, POSOCO simulation nadir 49.55 Hz and
  quasi-steady 49.71 Hz; 28 May 2020, 5,346 MW, drop 0.48 Hz (demand not reported, shown for a range).
Grid share at event A: not in our data (starts Sep 2021); proxy = median synchronous share of
  January 15:00-16:59 hours in 2022 (the earliest January we have).

Output: outputs/grid_cal_params.json, outputs/grid_cal_fit.csv, figures/grid_cal_check.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "outputs", ROOT / "figures"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

F0 = 50.0
TG, TT = 0.8, 10.0
HV = 2.0
DT, T_END = 0.005, 60.0


def simulate(dP, h, k, d, p_cap=None, r=None, e_avail=None, battery=False, t_end=T_END, keep=()):
    """Vectorised over cases. dP, h, k, d (and battery arrays) have the same length.
    Returns metrics dict (frequency in Hz) and traces for indices in keep."""
    n = len(dP)
    w = np.zeros(n); xg = np.zeros(n); pm = np.zeros(n); e_used = np.zeros(n)
    nadir = np.zeros(n); t_nadir = np.zeros(n); sat_t = np.zeros(n); p_peak = np.zeros(n)
    e_peak = np.zeros(n); e_hit = np.zeros(n, bool)
    cap = p_cap / r if battery else None
    win = int(round(0.5 / DT)); hist = np.zeros((win + 1, n)); rocof500 = np.zeros(n)
    f_final = -dP / (k + d); band = 0.05 * np.abs(f_final); last_out = np.zeros(n)
    traces = {i: [] for i in keep}
    steps = int(round(t_end / DT))
    for s in range(steps):
        a = pm - dP - d * w
        if battery:
            dw_free = a / (2 * (h + HV))
            p_req = -2 * HV * dw_free
            up = np.where(e_used < e_avail, cap, 0.0)
            p_bat = np.clip(p_req, -cap, up)
            clipped = np.abs(p_bat - p_req) > 1e-12
            dwdt = np.where(clipped, (a + p_bat) / (2 * h), dw_free)
            sat_t += (np.abs(p_req) > cap) * (up > 0) * DT
            e_hit |= (e_used >= e_avail) & (p_req > 0)
        else:
            p_bat = 0.0
            dwdt = a / (2 * h)
        if s == 0:
            rocof0 = np.abs(dwdt) * F0
        for i in keep:
            traces[i].append((s * DT, w[i] * F0, (p_bat[i] if battery else 0.0), pm[i]))
        w = w + dwdt * DT
        xg = xg + (-k * w - xg) / TG * DT
        pm = pm + (xg - pm) / TT * DT
        if battery:
            e_used = e_used + p_bat * DT
            e_peak = np.maximum(e_peak, e_used)
            p_peak = np.maximum(p_peak, p_bat)
        lower = w < nadir
        t_nadir = np.where(lower, (s + 1) * DT, t_nadir)
        nadir = np.minimum(nadir, w)
        hist[s % (win + 1)] = w
        if s >= win:
            rocof500 = np.maximum(rocof500, np.abs(w - hist[(s - win) % (win + 1)]) / 0.5 * F0)
        last_out = np.where(np.abs(w - f_final) > band, (s + 1) * DT, last_out)
    res = dict(nadir_dev_Hz=nadir * F0, t_nadir_s=t_nadir, rocof_t0_Hz_s=rocof0, rocof_500ms_Hz_s=rocof500,
               settling_s=np.where(np.abs(w - f_final) <= band, last_out, np.nan), final_dev_Hz=w * F0)
    if battery:
        res.update(saturated=sat_t > 0, sat_time_s=sat_t, energy_limit_hit=e_hit, p_bat_peak=p_peak,
                   energy_used_peak=e_peak)
    return res, traces


def frc_mw_per_hz(k, d, demand_mw):
    return (k + d) * demand_mw / F0


def main():
    g = pd.read_csv(OUT / "grid_india_hourly_clean.csv", parse_dates=["time"])
    g = g[g.valid]
    jan = g[(g.time.dt.year == 2022) & (g.time.dt.month == 1) & g.time.dt.hour.isin([15, 16])]
    s_A = float(jan.sync_share.clip(upper=1).median())
    print(f"proxy synchronous share for event A (Jan 2022, 15-16 h, median of {len(jan)} hours): {s_A:.3f}")

    demA, lossA = 144482.0, 780.0
    dPA = lossA / demA
    target = dict(rocof=0.0167, drop=0.081, t_nadir=10.6, frc=15000.0)

    # inertia from the initial RoCoF alone (governor not active at t = 0): H = dP f0 / (2 RoCoF)
    H_A = dPA * F0 / (2 * target["rocof"])
    print(f"event A: dP = {dPA:.5f} pu, inertia implied by RoCoF = {H_A:.2f} s (POSOCO estimate 8.06 s)")

    # grid search: h_s from RoCoF window, k_s and d from nadir depth, timing and steady response
    h_grid = np.arange(7.5, 10.51, 0.25)
    k_grid = np.arange(1.0, 12.01, 0.25)
    d_grid = np.array([1.0, 1.5, 2.0, 2.5])          # 2-5 %/Hz = 1.0-2.5 pu/pu
    H_, K_, D_ = np.meshgrid(h_grid, k_grid, d_grid, indexing="ij")
    hs, ks, ds = H_.ravel(), K_.ravel(), D_.ravel()
    n = len(hs)
    res, _ = simulate(np.full(n, dPA), hs * s_A, ks * s_A, ds, t_end=40.0)
    fit = pd.DataFrame(dict(h_s=hs, k_s=ks, d=ds, H=hs * s_A, K=ks * s_A, rocof500=res["rocof_500ms_Hz_s"],
                            rocof0=res["rocof_t0_Hz_s"], nadir_drop=-res["nadir_dev_Hz"], t_nadir=res["t_nadir_s"]))
    fit["frc"] = frc_mw_per_hz(fit.K, fit.d, demA)
    fit["power_number"] = lossA / fit.nadir_drop
    fit["err"] = (((fit.rocof0 - target["rocof"]) / target["rocof"]) ** 2
                  + ((fit.nadir_drop - target["drop"]) / target["drop"]) ** 2
                  + ((fit.t_nadir - target["t_nadir"]) / target["t_nadir"]) ** 2
                  + 0.25 * ((fit.frc - target["frc"]) / target["frc"]) ** 2)
    fit = fit.sort_values("err")
    fit.to_csv(OUT / "grid_cal_fit.csv", index=False)
    pd.set_option("display.width", 250)
    print("\nbest 8 parameter sets (event A):")
    print(fit.head(8).round(4).to_string(index=False))
    best = fit.iloc[0]
    P = dict(h_s=float(best.h_s), k_s=float(best.k_s), d=float(best.d), Tg=TG, Tt=TT, Hv=HV, dt=DT,
             proxy_sync_event_A=s_A)
    print(f"\nchosen: h_s {P['h_s']} s per unit synchronous share, k_s {P['k_s']}, D {P['d']} "
          f"(= {P['d'] / F0 * 100:.1f} % per Hz)")

    # checks
    rA, tA = simulate(np.array([dPA]), np.array([P["h_s"] * s_A]), np.array([P["k_s"] * s_A]),
                      np.array([P["d"]]), keep=[0])
    print("\nCHECK event A (21 Jan 2019, 780 MW): model vs measured")
    print(f"  initial RoCoF {rA['rocof_t0_Hz_s'][0]:.4f} vs 0.0167 Hz/s; 500 ms RoCoF {rA['rocof_500ms_Hz_s'][0]:.4f}")
    print(f"  nadir drop {-rA['nadir_dev_Hz'][0]:.4f} vs 0.081 Hz; time to nadir {rA['t_nadir_s'][0]:.1f} vs 10.6 s")
    print(f"  steady response {frc_mw_per_hz(P['k_s'] * s_A, P['d'], demA):.0f} MW/Hz (POSOCO median 15,000)")

    # typical recent hours: demand and synchronous share
    rec = g[g.time.dt.year == 2021]
    dem_med, s_med = float(rec.demand_MW.median()), float(rec.sync_share.clip(upper=1).median())
    rows = []
    for name, loss, dem, s in [("reference contingency 4,500 MW, 2021 median hour", 4500, dem_med, s_med),
                               ("28 May 2020, 5,346 MW, demand 150 GW", 5346, 150000, s_med),
                               ("28 May 2020, 5,346 MW, demand 170 GW", 5346, 170000, s_med),
                               ("28 May 2020, 5,346 MW, demand 190 GW", 5346, 190000, s_med)]:
        r, _ = simulate(np.array([loss / dem]), np.array([P["h_s"] * s]), np.array([P["k_s"] * s]),
                        np.array([P["d"]]))
        rows.append(dict(case=name, demand_MW=dem, sync=s, H=P["h_s"] * s, drop_Hz=-r["nadir_dev_Hz"][0],
                         t_nadir_s=r["t_nadir_s"][0], quasi_steady_drop_Hz=-r["final_dev_Hz"][0],
                         rocof500=r["rocof_500ms_Hz_s"][0],
                         frc_MW_Hz=frc_mw_per_hz(P["k_s"] * s, P["d"], dem)))
    V = pd.DataFrame(rows)
    print("\nCHECKS on other cases (reference: 4,500 MW -> nadir 49.55 Hz, quasi-steady 49.71 Hz in POSOCO "
          "simulation; 28 May 2020 -> drop 0.48 Hz measured; time to nadir 9-14 s):")
    print(V.round(3).to_string(index=False))
    P.update(check_event_A=dict(rocof0=float(rA["rocof_t0_Hz_s"][0]), drop=float(-rA["nadir_dev_Hz"][0]),
                                t_nadir=float(rA["t_nadir_s"][0])),
             checks=V.round(4).to_dict(orient="records"))
    json.dump(P, open(OUT / "grid_cal_params.json", "w"), indent=1)

    # figure: event A trace, model vs measured points; old model for comparison
    import power_sim as ps
    old, otr = ps.simulate(np.array([dPA]), np.ones(1), np.ones(1), np.ones(1), np.ones(1), battery=False,
                           keep=[0], h_sys=np.array([5.0 * s_A]), k_gov=np.array([s_A / 0.05]))
    t = np.array(tA[0]); to = np.array(otr[0])
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    ax[0].plot(t[:, 0], 49.885 + t[:, 1], label="calibrated model")
    ax[0].plot(to[:, 0], 49.885 + to[:, 1], "--", color="grey", label="old model (Task 3 settings)")
    ax[0].plot([10.6], [49.804], "r*", ms=14, label="measured nadir (POSOCO): 49.804 Hz at 10.6 s")
    ax[0].plot([0, 2], [49.885, 49.885 - 2 * 0.0167], "r:", lw=2, label="measured initial RoCoF 0.0167 Hz/s")
    ax[0].set(xlim=(0, 40), xlabel="time (s)", ylabel="frequency (Hz)",
              title="21 Jan 2019, 780 MW lost: model vs real event")
    ax[0].legend(fontsize=8)
    for _, row in V.iterrows():
        r, tr = simulate(np.array([row.demand_MW and (4500 if "4,500" in row.case else 5346) / row.demand_MW]),
                         np.array([row.H]), np.array([P["k_s"] * row.sync]), np.array([P["d"]]), keep=[0])
        tt = np.array(tr[0])
        ax[1].plot(tt[:, 0], 50 + tt[:, 1], label=row.case)
    ax[1].axhline(49.55, color="k", ls=":", lw=1, label="POSOCO simulated nadir, 4,500 MW (49.55)")
    ax[1].axhline(50 - 0.48, color="r", ls=":", lw=1, label="28 May 2020 measured drop (0.48 Hz)")
    ax[1].set(xlim=(0, 60), xlabel="time (s)", ylabel="frequency (Hz, from 50)", title="Checks on large events")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "grid_cal_check.png", dpi=150)
    print("saved outputs/grid_cal_params.json, outputs/grid_cal_fit.csv, figures/grid_cal_check.png")


if __name__ == "__main__":
    main()
