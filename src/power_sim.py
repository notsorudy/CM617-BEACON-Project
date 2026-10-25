"""
BEACON Task 3: SoH-dependent battery power cap and a single-area swing equation simulation.

1. Resistance curve R(SoH)/R(SoH=1), x = 1 - SoH_Q:
   quadratic   (main)       fitted on our training cells (outputs/resistance_curve_coefs.json)
   linear      (comparison) 1 + alpha*x, alpha = 0.84 fitted on our training cells
   forest      (check)      Model 2 random forest on SoH only, defined only where we have data
   lit_+30%, lit_+100%      NOT FROM OUR DATA: linear curves that reach +30% / +100% of R0
                            at SoH = 0.8 (literature end-of-life resistance rise)
   Each curve is divided by its own value at SoH = 1, so every curve starts at exactly 1.
   Our data covers SoH 0.76 to 1.0. Below 0.76 every curve is an extrapolation (flagged).
2. Power limit P_max = V_min*(V_oc - V_min)/R, so P_max/P_max_fresh = R(1)/R(SoH).
   Energy limit: reserve energy = T_RES seconds of fresh rated power, times SoH_Q.
3. Swing equation, per unit on the system base:
   2H dw/dt = Pm - dP_loss - D*w + P_bat
   Pm: governor of the remaining synchronous plants, first-order lag, droop.
   P_bat = -2*Hv*dw/dt (virtual inertia, Hv fixed), clipped to +/- P_cap(SoH) and to the
   energy left. H, Hv and the DC-link capacitor do NOT change with SoH.

Outputs: outputs/task3_curves.csv, outputs/task3_cases.csv, outputs/task3_sweep_soh.csv,
outputs/task3_sweep_demand.csv, outputs/task3_params.json, figures/task3_*.png
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "outputs", ROOT / "figures"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

# ---- cell values ----
V_OC, V_MIN = 3.30, 2.00          # V, LFP open-circuit (mid SOC) and minimum voltage
SOH_DATA_MIN = 0.76                # lowest SoH_Q backed by our data
SOH_EOL_LIT = 0.80                 # SoH where the literature rise is reached

# ---- grid values (per unit on system base) ----
F0 = 50.0                          # Hz
RE_SHARE = 0.60                    # inverter-based renewables (no inertia, no governor)
H_SYNC = 5.0                       # s, inertia of the synchronous plants on their own base
H = H_SYNC * (1 - RE_SHARE)        # 2.0 s on system base (low-inertia grid)
D = 1.0                            # load damping, pu power per pu frequency
DROOP = 0.05                       # 5% droop on synchronous capacity
K_GOV = (1 - RE_SHARE) / DROOP     # 8 pu/pu on system base
T_GOV = 3.0                        # s, governor + turbine lumped first-order lag
HV = 2.0                           # s, virtual inertia setting of the GFM inverter (fixed)
EVENTS = [0.05, 0.10]              # generation loss, pu of system load
DESIGN_EVENT = 0.10                # battery is sized against this event
DEMAND = [0.5, 0.8, 1.0, 1.2]      # peak fresh request / fresh P_max for the design event
T_RES = [10.0, 1.0]                # s of fresh rated power reserved for inertia (main, tight)
SOH_CASES = [1.0, 0.9, 0.82, 0.7]
DT, T_END = 0.001, 60.0
SETTLE_BAND = 0.05                 # settled when within 5% of the final deviation

coefs = json.load(open(OUT / "resistance_curve_coefs.json"))
ALPHA = coefs["alpha_formula"]
QA, QB, QC = coefs["quadratic_a_b_c"]
FOREST = joblib.load(OUT / "models" / "M2_soh__forest.joblib")
_forest_at_1 = float(FOREST.predict(pd.DataFrame({"SoH_Q": [1.0]}))[0])


def r_curve(name, soh):
    """R(SoH)/R(1). NaN where a curve is not defined."""
    soh = np.asarray(soh, dtype=float)
    x = 1 - soh
    if name == "quadratic":
        return (QA + QB * x + QC * x ** 2) / QA
    if name == "linear":
        return 1 + ALPHA * x
    if name == "forest":
        r = FOREST.predict(pd.DataFrame({"SoH_Q": np.atleast_1d(soh)})) / _forest_at_1
        return np.where(np.atleast_1d(soh) >= SOH_DATA_MIN, r, np.nan).reshape(soh.shape)
    if name == "lit_+30%":
        return 1 + 0.30 * x / (1 - SOH_EOL_LIT)
    if name == "lit_+100%":
        return 1 + 1.00 * x / (1 - SOH_EOL_LIT)
    raise ValueError(name)


CURVES = ["quadratic", "linear", "forest", "lit_+30%", "lit_+100%"]
LABEL = {"quadratic": "quadratic fit (main, our data)", "linear": f"linear formula, alpha = {ALPHA:.2f} (our data)",
         "forest": "random forest, SoH only (our data)", "lit_+30%": "literature +30% at EOL (NOT our data)",
         "lit_+100%": "literature +100% at EOL (NOT our data)"}
COLOR = {"quadratic": "tab:blue", "linear": "tab:red", "forest": "tab:green", "lit_+30%": "tab:orange",
         "lit_+100%": "tab:purple"}


def simulate(dP, p_cap0, r, soh, t_res, battery=True, keep=(), h_sys=None, k_gov=None):
    """Vectorised over cases. All inputs are arrays of the same length.
    h_sys, k_gov: per-case system inertia and governor gain (default: the fixed Task 3 values).
    Returns a dict of metric arrays, and traces for the case indices in keep."""
    n = len(dP)
    Hc = np.full(n, H) if h_sys is None else np.asarray(h_sys, dtype=float)
    Kc = np.full(n, K_GOV) if k_gov is None else np.asarray(k_gov, dtype=float)
    w = np.zeros(n)                # frequency deviation, pu
    pm = np.zeros(n)               # governor power change, pu
    e_used = np.zeros(n)           # net energy delivered by the battery, pu*s
    cap = p_cap0 / r
    e_avail = t_res * p_cap0 * soh
    steps = int(round(T_END / DT))
    nadir = np.zeros(n)
    rocof0 = np.zeros(n)
    sat_time = np.zeros(n)
    e_peak = np.zeros(n)
    p_peak = np.zeros(n)
    e_hit = np.zeros(n, dtype=bool)
    win = int(round(0.5 / DT))
    hist = np.zeros((win + 1, n))  # ring buffer for the 500 ms RoCoF
    rocof500 = np.zeros(n)
    f_final = -dP / (Kc + D)       # pu, analytical steady state (no secondary control)
    band = SETTLE_BAND * np.abs(f_final)
    last_out = np.zeros(n)
    traces = {i: [] for i in keep}
    for k in range(steps):
        a = pm - dP - D * w
        if battery:
            dwdt_free = a / (2 * (Hc + HV))
            p_req = -2 * HV * dwdt_free
            up = np.where(e_used < e_avail, cap, 0.0)
            p_bat = np.clip(p_req, -cap, up)
            clipped = np.abs(p_bat - p_req) > 1e-12
            dwdt = np.where(clipped, (a + p_bat) / (2 * Hc), dwdt_free)
            # power-cap saturation only (the energy cut-off is tracked separately)
            sat_time += (np.abs(p_req) > cap) * (up > 0) * DT
            e_hit |= (e_used >= e_avail) & (p_req > 0)
        else:
            p_bat = np.zeros(n)
            dwdt = a / (2 * Hc)
        if k == 0:
            rocof0 = np.abs(dwdt) * F0
        for i in keep:
            traces[i].append((k * DT, w[i] * F0, p_bat[i], cap[i], pm[i]))
        w = w + dwdt * DT
        pm = pm + (-Kc * w - pm) / T_GOV * DT
        e_used = e_used + p_bat * DT
        e_peak = np.maximum(e_peak, e_used)
        p_peak = np.maximum(p_peak, p_bat)
        nadir = np.minimum(nadir, w)
        hist[k % (win + 1)] = w
        if k >= win:
            rocof500 = np.maximum(rocof500, np.abs(w - hist[(k - win) % (win + 1)]) / 0.5 * F0)
        last_out = np.where(np.abs(w - f_final) > band, (k + 1) * DT, last_out)
    settled = np.abs(w - f_final) <= band
    return dict(nadir_Hz=nadir * F0, f_nadir_Hz=F0 + nadir * F0, rocof_t0_Hz_s=rocof0,
                rocof_500ms_Hz_s=rocof500, settling_s=np.where(settled, last_out, np.nan),
                saturated=sat_time > 0, sat_time_s=sat_time, p_bat_peak_pu=p_peak, p_cap_pu=cap,
                energy_used_peak_pu_s=e_peak, energy_avail_pu_s=e_avail, energy_limit_hit=e_hit,
                final_dev_Hz=w * F0), traces


def design_cap(d):
    """Fresh P_max (pu) so that the fresh peak inertia request in the design event equals d * P_max."""
    return HV / (H + HV) * DESIGN_EVENT / d


def run_grid(rows):
    df = pd.DataFrame(rows)
    res, _ = simulate(df.dP.to_numpy(), df.p_cap0.to_numpy(), df.r.to_numpy(), df.soh.to_numpy(),
                      df.t_res.to_numpy())
    for k, v in res.items():
        df[k] = v
    df["demand_actual"] = HV / (H + HV) * df.dP / df.p_cap0
    df["demand_vs_aged_cap"] = df.demand_actual * df.r
    df["extrapolated"] = df.soh < SOH_DATA_MIN
    return df


def main():
    params = dict(V_oc=V_OC, V_min=V_MIN, f0=F0, RE_share=RE_SHARE, H_sync=H_SYNC, H_system=H, D=D, droop=DROOP,
                  K_gov=K_GOV, T_gov=T_GOV, Hv=HV, events=EVENTS, design_event=DESIGN_EVENT, demand=DEMAND,
                  T_res=T_RES, dt=DT, t_end=T_END, settle_band=SETTLE_BAND, soh_data_min=SOH_DATA_MIN,
                  alpha=ALPHA, quadratic_a_b_c=[QA, QB, QC])
    json.dump(params, open(OUT / "task3_params.json", "w"), indent=1)

    # ---- 1. curves ----
    soh = np.round(np.arange(0.60, 1.0001, 0.005), 3)
    cv = pd.DataFrame({"SoH": soh, "extrapolated": soh < SOH_DATA_MIN})
    for c in CURVES:
        cv[f"r_{c}"] = r_curve(c, soh)
        cv[f"Pmax_frac_{c}"] = 1 / cv[f"r_{c}"]
    cv["E_frac"] = soh
    cv.to_csv(OUT / "task3_curves.csv", index=False)
    inr = cv[~cv.extrapolated]
    print("Curve check against the forest (SoH 0.76 to 1.0):")
    for c in ["quadratic", "linear", "lit_+30%", "lit_+100%"]:
        d = inr[f"r_{c}"] - inr["r_forest"]
        print(f"  {c:10s} RMSE vs forest {np.sqrt((d ** 2).mean()):.4f}, max abs diff {d.abs().max():.4f}")
    print("\nR/R0 and P_max/P_max_fresh at the case SoH values:")
    for s in SOH_CASES:
        txt = " | ".join(f"{c} r={float(r_curve(c, s)):.3f} P={1 / float(r_curve(c, s)):.3f}" for c in CURVES)
        print(f"  SoH {s:.2f}{' (extrapolated)' if s < SOH_DATA_MIN else ''}: {txt}")

    tab = pd.read_csv(OUT / "cycle_table.csv")
    r0_cell = tab.groupby("cell_id").R0_ohm.first()
    r0 = float(r0_cell.median())
    p_cell = V_MIN * (V_OC - V_MIN) / r0
    print(f"\nCell: median R0 = {r0 * 1000:.2f} mOhm (124 cells, range {r0_cell.min() * 1000:.2f} to "
          f"{r0_cell.max() * 1000:.2f}); fresh P_max = {p_cell:.1f} W at I = {(V_OC - V_MIN) / r0:.1f} A")
    for s in [0.9, 0.82, 0.7]:
        print(f"  SoH {s}: P_max quadratic {p_cell / float(r_curve('quadratic', s)):.1f} W, "
              f"linear {p_cell / float(r_curve('linear', s)):.1f} W")
    params.update(R0_median_ohm=r0, P_max_cell_fresh_W=p_cell)
    json.dump(params, open(OUT / "task3_params.json", "w"), indent=1)

    # ---- 2. discrete cases ----
    rows = []
    for ev in EVENTS:
        for d in DEMAND:
            for tr in T_RES:
                for c in CURVES:
                    for s in SOH_CASES:
                        r = float(r_curve(c, s))
                        if np.isnan(r):
                            continue
                        rows.append(dict(event=ev, dP=ev, design_demand=d, p_cap0=design_cap(d), t_res=tr,
                                         curve=c, soh=s, r=r))
    cases = run_grid(rows)
    # no-battery and uncapped-battery references
    ref_nb, _ = simulate(np.array(EVENTS), np.ones(2), np.ones(2), np.ones(2), np.full(2, 1e9), battery=False)
    ref_ideal, _ = simulate(np.array(EVENTS), np.full(2, 1e9), np.ones(2), np.ones(2), np.full(2, 1e9))
    refs = pd.DataFrame([dict(event=ev, ref=n, nadir_Hz=r["nadir_Hz"][i], rocof_t0=r["rocof_t0_Hz_s"][i],
                              rocof_500ms=r["rocof_500ms_Hz_s"][i], settling_s=r["settling_s"][i],
                              final_dev_Hz=r["final_dev_Hz"][i])
                         for n, r in [("no battery", ref_nb), ("uncapped battery", ref_ideal)]
                         for i, ev in enumerate(EVENTS)])
    refs.to_csv(OUT / "task3_references.csv", index=False)
    print("\nReferences:")
    print(refs.round(4).to_string(index=False))

    fresh = cases[cases.soh == 1.0].set_index(["event", "design_demand", "t_res", "curve"])
    key = ["event", "design_demand", "t_res", "curve"]
    f = fresh.reindex(pd.MultiIndex.from_frame(cases[key]))
    cases["d_nadir_vs_fresh_Hz"] = cases.nadir_Hz.to_numpy() - f.nadir_Hz.to_numpy()
    cases["d_rocof500_vs_fresh"] = cases.rocof_500ms_Hz_s.to_numpy() - f.rocof_500ms_Hz_s.to_numpy()
    cases["d_settling_vs_fresh_s"] = cases.settling_s.to_numpy() - f.settling_s.to_numpy()
    cases["aging_no_difference"] = ((cases.d_nadir_vs_fresh_Hz.abs() < 1e-3)
                                    & (cases.d_rocof500_vs_fresh.abs() < 1e-3))
    cases.to_csv(OUT / "task3_cases.csv", index=False)

    show = ["event", "design_demand", "demand_actual", "t_res", "curve", "soh", "r", "demand_vs_aged_cap", "f_nadir_Hz",
            "rocof_t0_Hz_s", "rocof_500ms_Hz_s", "settling_s", "saturated", "sat_time_s", "energy_limit_hit",
            "d_nadir_vs_fresh_Hz", "aging_no_difference"]
    pd.set_option("display.width", 250)
    for tr in T_RES:
        print(f"\n=== cases, energy reserve {tr} s, main curves ===")
        sub = cases[(cases.t_res == tr) & cases.curve.isin(["quadratic", "linear", "lit_+100%"])]
        print(sub[show].round(4).to_string(index=False))

    # ---- 3. sweeps for figures (energy reserve 10 s) ----
    rows = []
    soh_grid = np.round(np.arange(0.60, 1.0001, 0.01), 3)
    for ev in EVENTS:
        for d in DEMAND:
            for c in CURVES:
                for s in soh_grid:
                    r = float(r_curve(c, s))
                    if not np.isnan(r):
                        rows.append(dict(event=ev, dP=ev, design_demand=d, p_cap0=design_cap(d), t_res=10.0,
                                         curve=c, soh=s, r=r))
    sw = run_grid(rows)
    sw.to_csv(OUT / "task3_sweep_soh.csv", index=False)

    rows = []
    for d in np.round(np.arange(0.3, 1.61, 0.05), 3):
        for c in ["quadratic", "linear", "lit_+100%"]:
            for s in SOH_CASES:
                rows.append(dict(event=0.10, dP=0.10, design_demand=d, p_cap0=design_cap(d), t_res=10.0,
                                 curve=c, soh=s, r=float(r_curve(c, s))))
    sd = run_grid(rows)
    sd.to_csv(OUT / "task3_sweep_demand.csv", index=False)

    # ---- 4. traces for the frequency response figure ----
    tr_rows = [dict(soh=s, curve=c) for c in ["quadratic", "lit_+100%"] for s in SOH_CASES]
    n = len(tr_rows)
    _, traces = simulate(np.full(n, 0.10), np.full(n, design_cap(1.0)),
                         np.array([float(r_curve(x["curve"], x["soh"])) for x in tr_rows]),
                         np.array([x["soh"] for x in tr_rows]), np.full(n, 10.0), keep=range(n))
    _, tr_nb = simulate(np.array([0.10]), np.ones(1), np.ones(1), np.ones(1), np.ones(1), battery=False, keep=[0])

    make_figures(cv, sw, sd, tr_rows, traces, tr_nb[0])
    print("\nsaved outputs and figures")


def shade(ax, lo=0.6):
    ax.axvspan(lo, SOH_DATA_MIN, color="grey", alpha=0.18, lw=0)
    ax.text(SOH_DATA_MIN - 0.005, 0.97, "extrapolation\n(no data)", transform=ax.get_xaxis_transform(), va="top",
            ha="left", fontsize=8)


def make_figures(cv, sw, sd, tr_rows, traces, tr_nb):
    # R vs SoH and P_max vs SoH
    bins = pd.read_csv(OUT / "predictions.csv")[["SoH_Q", "R_ratio"]]
    b = bins.groupby(pd.cut(bins.SoH_Q, np.arange(0.76, 1.02, 0.02)), observed=True).agg(S=("SoH_Q", "mean"),
                                                                                       R=("R_ratio", "mean"))
    for fname, col, ylab in [("task3_R_vs_SoH.png", "r_", "R / R(SoH = 1)"),
                             ("task3_Pmax_vs_SoH.png", "Pmax_frac_", "P_max / P_max fresh")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for c in CURVES:
            ax.plot(cv.SoH, cv[col + c], color=COLOR[c], lw=2, ls="--" if c.startswith("lit") else "-",
                    label=LABEL[c])
        if col == "r_":
            ax.plot(b.S, b.R, "ko", ms=4, label="data, binned mean (all cells)")
        else:
            ax.plot(cv.SoH, cv.E_frac, color="k", lw=1.5, ls=":", label="energy limit E / E fresh = SoH_Q")
        shade(ax)
        ax.set(xlabel="SoH_Q", ylabel=ylab, xlim=(0.6, 1.0))
        ax.invert_xaxis()
        ax.legend(fontsize=8, loc="upper left" if col == "r_" else "lower left")
        fig.tight_layout()
        fig.savefig(FIG / fname, dpi=150)
        plt.close(fig)

    # frequency response fresh vs aged
    fig, ax = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    t_nb = np.array(tr_nb)
    for j, c in enumerate(["quadratic", "lit_+100%"]):
        ax[0, j].plot(t_nb[:, 0], F0 + t_nb[:, 1], color="grey", lw=1.5, label="no battery")
        for i, row in enumerate(tr_rows):
            if row["curve"] != c:
                continue
            t = np.array(traces[i])
            lab = f"SoH {row['soh']}" + (" (extrapolated)" if row["soh"] < SOH_DATA_MIN else "")
            ls = ":" if row["soh"] < SOH_DATA_MIN else "-"
            ax[0, j].plot(t[:, 0], F0 + t[:, 1], ls=ls, lw=1.5, label=lab)
            l, = ax[1, j].plot(t[:, 0], t[:, 2], ls=ls, lw=1.5, label=lab)
            ax[1, j].axhline(t[0, 3], color=l.get_color(), ls="--", lw=0.8)
        ax[0, j].set(ylabel="frequency (Hz)")
        ax[0, j].set_title(f"10% generation loss, demand ratio 1.0\nR(SoH): {LABEL[c]}", fontsize=11)
        ax[1, j].set(xlabel="time (s)", ylabel="battery power (pu of system load)",
                     title="battery power (dashed: cap P_max(SoH))")
        ax[0, j].legend(fontsize=8)
        ax[0, j].set_xlim(0, 20)
    fig.tight_layout()
    fig.savefig(FIG / "task3_frequency_response.png", dpi=150)
    plt.close(fig)

    # nadir and RoCoF vs SoH, per event
    for ev in EVENTS:
        fig, ax = plt.subplots(2, len(DEMAND), figsize=(17, 7.5), sharey="row")
        for j, d in enumerate(DEMAND):
            s = sw[(sw.event == ev) & (sw.design_demand == d)]
            for c in CURVES:
                x = s[s.curve == c]
                ax[0, j].plot(x.soh, x.f_nadir_Hz, color=COLOR[c], ls="--" if c.startswith("lit") else "-",
                              label=LABEL[c])
                ax[1, j].plot(x.soh, x.rocof_500ms_Hz_s, color=COLOR[c], ls="--" if c.startswith("lit") else "-")
            da = s.demand_actual.iloc[0]
            ax[0, j].set_title(f"demand ratio {da:.2f} (battery sized at {d} for 10% loss)", fontsize=10)
            for a in ax[:, j]:
                shade(a)
                a.set_xlim(0.6, 1.0)
                a.invert_xaxis()
            ax[1, j].set_xlabel("SoH_Q")
        ax[0, 0].set_ylabel("frequency nadir (Hz)")
        ax[1, 0].set_ylabel("RoCoF, 500 ms window (Hz/s)")
        ax[0, 0].legend(fontsize=7, loc="lower left")
        fig.suptitle(f"{int(ev * 100)}% generation loss, energy reserve 10 s")
        fig.tight_layout()
        fig.savefig(FIG / f"task3_nadir_rocof_vs_SoH_{int(ev * 100)}pct.png", dpi=150)
        plt.close(fig)

    # nadir and RoCoF vs demand ratio
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for c, ls in [("quadratic", "-"), ("linear", "-."), ("lit_+100%", "--")]:
        for s, col in zip(SOH_CASES, ["k", "tab:blue", "tab:orange", "tab:red"]):
            x = sd[(sd.curve == c) & (sd.soh == s)]
            lab = f"{c}, SoH {s}" + (" (extrap.)" if s < SOH_DATA_MIN else "")
            if c != "quadratic" and s == 1.0:
                continue  # fresh is the same for every curve
            ax[0].plot(x.demand_actual, x.f_nadir_Hz, ls=ls, color=col, label=lab)
            ax[1].plot(x.demand_actual, x.rocof_500ms_Hz_s, ls=ls, color=col)
    ax[0].set(xlabel="demand ratio = peak inertia request / fresh P_max", ylabel="frequency nadir (Hz)",
              title="10% generation loss, energy reserve 10 s")
    ax[1].set(xlabel="demand ratio = peak inertia request / fresh P_max", ylabel="RoCoF, 500 ms window (Hz/s)")
    ax[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "task3_nadir_rocof_vs_demand.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
