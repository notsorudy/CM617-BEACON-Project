"""
BEACON: synthetic inertia from a fresh vs aged battery, over the real Indian grid hours
(Grid-India SCADA, Mendeley 10.17632/y58jknpgs8.2, cleaned in clean_grid_data.py).

For every clean hour:
  sync_share(t) = real synchronous generation / demand (to 2023), 1 - (wind+solar)/demand (2024-25),
                  capped at 1.0
  H(t)     = H_SYNC * sync_share(t)          (H_SYNC = 5 s, assumption; the data gives MW, not H)
  K_gov(t) = sync_share(t) / DROOP           (only synchronous plants give governor response)
Battery: virtual inertia Hv = 2 s, cap = fresh P_max / r(SoH), energy reserve 10 s x SoH.
Sizing: fresh P_max set against the lowest-inertia clean hour in the data, for a 10% loss,
  so that (peak inertia request in that hour) / (fresh P_max) = d, d in {0.8, 1.0, 1.2}.
  The same battery is used in every hour and for the 5% event.
Speed: the result of an hour depends only on sync_share, so sync_share is rounded to 0.005 and each
  distinct value is simulated once, then mapped back to all hours.

Outputs: outputs/real_grid_bins.csv (one row per simulated case), outputs/real_grid_hourly.csv
(main metrics per hour), outputs/real_grid_summary.csv (per year), figures/real_grid_*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import power_sim as ps

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "outputs", ROOT / "figures"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
pd.set_option("display.width", 250)

BIN = 0.005
SOHS = [1.0, 0.9, 0.82, 0.7]
CURVES = ["quadratic", "linear", "lit_+100%"]
DEMANDS = [0.8, 1.0, 1.2]
EVENTS = [0.05, 0.10]
T_RES = 10.0
GRID_INDIA_H = (5.0, 9.0)   # s, Grid-India NASPI 2023 slides (not yet checked in the full report)


def main():
    # regression check: defaults must reproduce Task 3
    ref = pd.read_csv(OUT / "task3_references.csv")
    nb, _ = ps.simulate(np.array([0.05, 0.10]), np.ones(2), np.ones(2), np.ones(2), np.full(2, 1e9), battery=False)
    old = ref[ref.ref == "no battery"].sort_values("event").nadir_Hz.to_numpy()
    print(f"regression check, no-battery nadir now {nb['nadir_Hz'].round(4)} vs Task 3 {old.round(4)}")
    assert np.allclose(nb["nadir_Hz"], old, atol=1e-6)

    g = pd.read_csv(OUT / "grid_india_hourly_clean.csv", parse_dates=["time"])
    g = g[g.valid].copy()
    g["sync"] = g.sync_share.clip(upper=1.0)
    g["sync_bin"] = (np.round(g.sync / BIN) * BIN).round(4)
    g["H_s"] = ps.H_SYNC * g.sync
    g["year"] = g.time.dt.year
    print(f"clean hours {len(g)}; sync share capped at 1.0 in {int((g.sync_share > 1).sum())} hours")
    print(f"distinct sync bins {g.sync_bin.nunique()}, max rounding error {(g.sync - g.sync_bin).abs().max():.4f}")
    print("\nsystem inertia H = 5 s x sync share, by year (s):")
    print(g.groupby("year").H_s.describe(percentiles=[.01, .05, .5]).round(2).to_string())

    worst = g.loc[g.sync.idxmin()]
    s_min = float(worst.sync_bin)
    h_min = ps.H_SYNC * s_min
    peak_req = ps.HV / (h_min + ps.HV) * 0.10
    print(f"\nsizing hour: {worst.time} sync {worst.sync:.4f} (bin {s_min}), H {h_min:.3f} s, "
          f"peak inertia request for 10% loss {peak_req:.4f} pu")

    # all cases on the distinct bins
    bins = np.sort(g.sync_bin.unique())
    rows = []
    for ev in EVENTS:
        for d in DEMANDS:
            for c in CURVES:
                for s in SOHS:
                    for b in bins:
                        rows.append(dict(event=ev, design_demand=d, curve=c, soh=s, sync_bin=b,
                                         r=float(ps.r_curve(c, s)), p_cap0=peak_req / d))
    for ev in EVENTS:   # no-battery reference
        for b in bins:
            rows.append(dict(event=ev, design_demand=np.nan, curve="no battery", soh=np.nan, sync_bin=b,
                             r=1.0, p_cap0=0.0))
    cs = pd.DataFrame(rows)
    nobat = cs.curve == "no battery"
    res = {}
    for flag, sub in [(False, cs[~nobat]), (True, cs[nobat])]:
        r, _ = ps.simulate(sub.event.to_numpy(), np.maximum(sub.p_cap0.to_numpy(), 1e-12), sub.r.to_numpy(),
                           sub.soh.fillna(1.0).to_numpy(), np.full(len(sub), T_RES), battery=not flag,
                           h_sys=ps.H_SYNC * sub.sync_bin.to_numpy(), k_gov=sub.sync_bin.to_numpy() / ps.DROOP)
        for k, v in r.items():
            res.setdefault(k, np.empty(len(cs), dtype=v.dtype))
            res[k][sub.index.to_numpy()] = v
    for k, v in res.items():
        cs[k] = v
    cs["H_s"] = ps.H_SYNC * cs.sync_bin
    cs["peak_request_pu"] = ps.HV / (cs.H_s + ps.HV) * cs.event
    cs["demand_this_hour"] = cs.peak_request_pu / cs.p_cap0.replace(0, np.nan)
    cs["extrapolated"] = cs.soh < ps.SOH_DATA_MIN
    cs.to_csv(OUT / "real_grid_bins.csv", index=False)
    print(f"simulated {len(cs)} cases")

    # map to hours and summarise
    key = ["event", "design_demand", "curve", "soh"]
    summ = []
    fresh = cs[cs.soh == 1.0].set_index(["event", "design_demand", "curve", "sync_bin"]).sort_index()
    for (ev, d, c, s), sub in cs[~nobat].groupby(key):
        m = g[["time", "year", "sync_bin"]].merge(sub, on="sync_bin", how="left")
        f = fresh.loc[(ev, d, c)].reindex(m.sync_bin)
        drocof = m.rocof_500ms_Hz_s.to_numpy() - f.rocof_500ms_Hz_s.to_numpy()
        dnadir = m.nadir_Hz.to_numpy() - f.nadir_Hz.to_numpy()
        m["drocof"], m["dnadir"] = drocof, dnadir
        for yr, y in list(m.groupby("year")) + [("all", m)]:
            summ.append(dict(event=ev, design_demand=d, curve=c, soh=s, year=yr, hours=len(y),
                             pct_hours_saturated=100 * y.saturated.mean(),
                             pct_hours_aging_matters=100 * ((y.drocof.abs() > 1e-3) | (y.dnadir.abs() > 1e-3)).mean(),
                             rocof_median=y.rocof_500ms_Hz_s.median(), rocof_max=y.rocof_500ms_Hz_s.max(),
                             drocof_max=y.drocof.max(), drocof_mean_when_matters=y.drocof[y.drocof > 1e-3].mean(),
                             nadir_min_Hz=y.f_nadir_Hz.min(), dnadir_min=y.dnadir.min(),
                             energy_limit_hit_pct=100 * y.energy_limit_hit.mean()))
    sm = pd.DataFrame(summ)
    sm.to_csv(OUT / "real_grid_summary.csv", index=False)

    nb = cs[nobat].set_index(["event", "sync_bin"])
    for ev in EVENTS:
        x = nb.loc[ev].reindex(g.sync_bin)
        print(f"\nno battery, {int(ev * 100)}% loss: nadir {x.f_nadir_Hz.min():.3f} to {x.f_nadir_Hz.max():.3f} Hz, "
              f"RoCoF 500 ms {x.rocof_500ms_Hz_s.min():.3f} to {x.rocof_500ms_Hz_s.max():.3f} Hz/s")

    show = ["event", "design_demand", "curve", "soh", "year", "hours", "pct_hours_saturated", "pct_hours_aging_matters",
            "rocof_median", "rocof_max", "drocof_max", "nadir_min_Hz", "dnadir_min", "energy_limit_hit_pct"]
    for ev in EVENTS:
        print(f"\n===== {int(ev * 100)}% loss, all years =====")
        print(sm[(sm.event == ev) & (sm.year == "all")][show].round(4).to_string(index=False))
    print("\n===== by year: quadratic, 10% loss =====")
    q = sm[(sm.event == 0.10) & (sm.curve == "quadratic") & (sm.year != "all") & (sm.soh < 1.0)]
    print(q.pivot_table(index=["design_demand", "soh"], columns="year", values="pct_hours_aging_matters")
          .round(2).to_string())
    print("\n===== by year: literature +100%, 10% loss =====")
    q = sm[(sm.event == 0.10) & (sm.curve == "lit_+100%") & (sm.year != "all") & (sm.soh < 1.0)]
    print(q.pivot_table(index=["design_demand", "soh"], columns="year", values="pct_hours_aging_matters")
          .round(2).to_string())

    # per-hour file for the main case (quadratic, d = 1.0)
    main_rows = cs[(cs.curve == "quadratic") & (cs.design_demand == 1.0)]
    hourly = g[["time", "year", "demand_MW", "wind_MW", "solar_MW", "vre_share", "sync", "sync_bin", "H_s"]].copy()
    for ev in EVENTS:
        for s in SOHS:
            x = main_rows[(main_rows.event == ev) & (main_rows.soh == s)].set_index("sync_bin").reindex(hourly.sync_bin)
            tag = f"{int(ev * 100)}pct_soh{s}"
            hourly[f"rocof_{tag}"] = x.rocof_500ms_Hz_s.to_numpy()
            hourly[f"nadir_{tag}"] = x.f_nadir_Hz.to_numpy()
            hourly[f"sat_{tag}"] = x.saturated.to_numpy()
        x = nb.loc[ev].reindex(hourly.sync_bin)
        hourly[f"rocof_{int(ev * 100)}pct_nobattery"] = x.rocof_500ms_Hz_s.to_numpy()
    hourly.to_csv(OUT / "real_grid_hourly.csv", index=False)
    json.dump(dict(H_sync=ps.H_SYNC, droop=ps.DROOP, D=ps.D, T_gov=ps.T_GOV, Hv=ps.HV, T_res=T_RES, bin=BIN,
                   sizing_hour=str(worst.time), sizing_sync=s_min, sizing_H=h_min, peak_request_10pct=peak_req,
                   demands=DEMANDS, sohs=SOHS, curves=CURVES, grid_india_reported_H=GRID_INDIA_H),
              open(OUT / "real_grid_params.json", "w"), indent=1)

    make_figures(g, cs, sm, hourly, h_min)
    print("\nsaved outputs and figures")


def make_figures(g, cs, sm, hourly, h_min):
    # 1. inertia over time vs Grid-India range
    fig, ax = plt.subplots(figsize=(11, 4.5))
    w = g.set_index("time").H_s
    ax.plot(w.resample("W").mean().index, w.resample("W").mean(), label="weekly mean")
    ax.plot(w.resample("W").min().index, w.resample("W").min(), lw=0.8, label="weekly lowest hour")
    ax.axhspan(*GRID_INDIA_H, color="tab:green", alpha=0.12, label="Grid-India reported 5 to 9 s (slides)")
    ax.axhline(h_min, color="k", ls="--", lw=1, label=f"battery sizing hour, H = {h_min:.2f} s")
    ax.set(ylabel="system inertia H (s)", title="Hourly inertia from real Grid-India data, H = 5 s x synchronous share",
           ylim=(2.5, 9.5))
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "real_grid_inertia.png", dpi=150)
    plt.close(fig)

    # 2. share of hours where aging matters, by year
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    years = sorted(y for y in sm.year.unique() if y != "all")
    for a, d in zip(ax, DEMANDS):
        for c, ls in [("quadratic", "-"), ("linear", "-."), ("lit_+100%", "--")]:
            for s, col in [(0.9, "tab:blue"), (0.82, "tab:orange"), (0.7, "tab:red")]:
                x = sm[(sm.event == 0.10) & (sm.design_demand == d) & (sm.curve == c) & (sm.soh == s)
                       & (sm.year != "all")].set_index("year").reindex(years)
                lab = f"{c}, SoH {s}" + (" (extrap.)" if s < ps.SOH_DATA_MIN else "")
                a.plot([str(y) for y in years], x.pct_hours_aging_matters, ls=ls, color=col, marker="o", ms=3,
                       label=lab)
        a.set(title=f"battery sized at demand ratio {d} for the worst hour", xlabel="year (2021: Sep-Dec, 2025: Jan-Jun)")
    ax[0].set_ylabel("% of hours where aging changes the response")
    ax[0].legend(fontsize=7)
    fig.suptitle("10% generation loss, real Grid-India hours")
    fig.tight_layout()
    fig.savefig(FIG / "real_grid_aging_hours_by_year.png", dpi=150)
    plt.close(fig)

    # 3. RoCoF vs synchronous share, fresh vs aged (main case)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    for a, ev in zip(ax, EVENTS):
        sub = cs[(cs.event == ev) & (cs.curve == "quadratic") & (cs.design_demand == 1.0)]
        nb = cs[(cs.event == ev) & (cs.curve == "no battery")].sort_values("sync_bin")
        a.plot(1 - nb.sync_bin, nb.rocof_500ms_Hz_s, color="grey", label="no battery")
        for s, col in [(1.0, "k"), (0.9, "tab:blue"), (0.82, "tab:orange"), (0.7, "tab:red")]:
            x = sub[sub.soh == s].sort_values("sync_bin")
            a.plot(1 - x.sync_bin, x.rocof_500ms_Hz_s, color=col, ls=":" if s < ps.SOH_DATA_MIN else "-",
                   label=f"SoH {s}" + (" (extrapolated)" if s < ps.SOH_DATA_MIN else ""))
        a.set(xlabel="non-synchronous share of demand (real hours span this range)", ylabel="RoCoF, 500 ms (Hz/s)",
              title=f"{int(ev * 100)}% loss, quadratic R(SoH), battery sized at demand ratio 1.0")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "real_grid_rocof_vs_share.png", dpi=150)
    plt.close(fig)

    # 4. average day in 2025: RoCoF fresh vs aged, and saturated hours by hour of day
    y = hourly[hourly.year == 2025]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    hod = y.time.dt.hour
    ax[0].plot(y.groupby(hod).rocof_10pct_nobattery.mean(), color="grey", label="no battery")
    for s, col in [(1.0, "k"), (0.9, "tab:blue"), (0.82, "tab:orange")]:
        ax[0].plot(y.groupby(hod)[f"rocof_10pct_soh{s}"].mean(), color=col, label=f"SoH {s}")
    ax[0].set(xlabel="hour of day (IST)", ylabel="mean RoCoF, 500 ms (Hz/s)",
              title="2025 (Jan-Jun), 10% loss, quadratic, demand ratio 1.0")
    ax[0].legend(fontsize=8)
    for s, col in [(0.9, "tab:blue"), (0.82, "tab:orange"), (0.7, "tab:red")]:
        ax[1].plot(100 * y.groupby(hod)[f"sat_10pct_soh{s}"].mean(), color=col, marker="o", ms=3,
                   ls=":" if s < ps.SOH_DATA_MIN else "-",
                   label=f"SoH {s}" + (" (extrapolated)" if s < ps.SOH_DATA_MIN else ""))
    ax[1].set(xlabel="hour of day (IST)", ylabel="% of 2025 hours with cap saturated",
              title="When does the aged battery hit its cap? (2025)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "real_grid_2025_by_hour.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
