"""
BEACON: frequency response over time for the solar-only / wind-only system (battery = only inertia).
Same system, battery states and real Kutch hours as re_only.py (main size 25 MW / 50 MWh, reserve 12.5 MW).

Model (MW and per-unit frequency deviation w; 50 Hz base), first 20 s after a sudden loss of plant output dP:
  grid-forming battery (virtual synchronous machine) asks for
      p_req = -2 Hv S dw/dt - Kd S w - Ki S z,   z = integral of w
      Hv = 4 s virtual inertia (as re_only.py), Kd = 1/droop with droop 1 % on the battery rating,
      Ki = Kd / 10 s (slow restoring control that brings frequency back to 50 Hz, so the battery ends up carrying
      the whole loss, as assumed in re_only.py)
  it can only add its spare power: head = power ceiling(SoH) - dispatch in that hour. If p_req > head the
  battery is at its current limit: it delivers head, its inertia and droop are gone, and frequency is held only
  by the load: motor inertia H_L (on the load) and load damping D = 1.5 (as re_only.py).
  load balance:  2 H_L L dw/dt = p_bat - dP - D L w,  L = plant output + dispatch (as re_only.py)
  H_L = 0.5 s is an ASSUMPTION (rotating motors in the load); sensitivity 0.2 s and 1.0 s.
  Load shedding (49.4 Hz) is flagged but not modelled as an action. Energy is not limiting in 20 s.
Usage: python re_freq.py
"""
import numpy as np
import pandas as pd

import power_sim as ps
from re_only import (OUT, PLANT_MW, BAT_MW, BAT_MWH, RESERVE_MW, SOC_DISPATCH_MIN, D_LOAD, UFLS_HZ, HV, F0,
                     MIN_GEN_MW, EVENTS_MW, load_kutch, dispatch, aware_soc_min)

DROOP = 0.01
KD = 1 / DROOP
KI = KD / 10.0
H_LOAD = 0.5
DT, T_END = 0.002, 20.0


def simulate(dP, L, head, s_mw, h_load=H_LOAD, keep=()):
    n = len(dP)
    w, z = np.zeros(n), np.zeros(n)
    m_load = 2 * h_load * L
    m_tot = m_load + 2 * HV * s_mw
    nadir, t_below, rocof500 = np.zeros(n), np.zeros(n), np.zeros(n)
    clipped_any = np.zeros(n, dtype=bool)
    win = int(round(0.5 / DT))
    hist = np.zeros((win + 1, n))
    traces = {i: [] for i in keep}
    for k in range(int(round(T_END / DT))):
        a = -dP - D_LOAD * L * w
        dwdt_free = (a - KD * s_mw * w - KI * s_mw * z) / m_tot
        p_req = -2 * HV * s_mw * dwdt_free - KD * s_mw * w - KI * s_mw * z
        clip = p_req > head
        p = np.where(clip, head, p_req)
        dwdt = np.where(clip, (head + a) / m_load, dwdt_free)
        if k == 0:
            rocof0 = np.abs(dwdt) * F0
        for i in keep:
            traces[i].append((k * DT, F0 * (1 + w[i]), p[i]))
        clipped_any |= clip
        z = z + np.where(clip, 0.0, w) * DT
        w = w + dwdt * DT
        nadir = np.minimum(nadir, w)
        t_below += (w * F0 < -UFLS_HZ) * DT
        hist[k % (win + 1)] = w
        if k >= win:
            rocof500 = np.maximum(rocof500, np.abs(w - hist[(k - win) % (win + 1)]) / 0.5 * F0)
    return dict(f_min_Hz=F0 * (1 + nadir), rocof0_Hz_s=rocof0, rocof500_Hz_s=rocof500, s_below_49p4=t_below,
                f_end_Hz=F0 * (1 + w), at_limit=clipped_any), traces


def cases():
    S = pd.read_csv(OUT / "re_soh.csv")
    S = S[S.temp_case == "site"]
    out = [("fresh", "SoH-unaware", 1.0)]
    for c in ["LFP", "NMC", "NCA"]:
        for mode in ["SoH-unaware", "SoH-aware"]:
            out.append((c, mode, None))
    return S, out


def battery_state(S, src, chem, mode, soh, gen):
    if soh is None:
        soh = float(S[(S.source == src) & (S.chem == chem) & (S.year == 10)].SoH.iloc[0])
    p_cap, e = BAT_MW / float(ps.r_curve("quadratic", soh)), BAT_MWH * soh
    p_lim = RESERVE_MW if mode == "SoH-unaware" else max(p_cap - RESERVE_MW, 0.0)
    p_lim = min(p_lim, p_cap)
    soc_min = SOC_DISPATCH_MIN if mode == "SoH-unaware" else aware_soc_min(p_lim, e)
    p, _ = dispatch(gen, p_lim, e, soc_min=soc_min)
    return soh, p_cap, p


def main():
    df = load_kutch()
    S, cs = cases()
    rows, ex_rows, ex_traces = [], [], []
    for src in ["solar", "wind"]:
        gen = PLANT_MW * df[f"{src}_cf"].to_numpy()
        act = gen >= MIN_GEN_MW
        states = {}
        for chem, mode, soh in cs:
            states[(chem, mode)] = battery_state(S, src, chem, mode, soh, gen)
        # example hour: an hour where the aged NCA battery (SoH-unaware) cannot cover the design trip,
        # the one with the median shortfall among those hours
        _, cap_nca, p_nca = states[("NCA", "SoH-unaware")]
        short = np.minimum(10.0, gen) - (cap_nca - p_nca)
        bad = np.where(act & (short > 0))[0]
        ex = int(bad[np.argsort(short[bad])[len(bad) // 2]])
        for ev, mw in EVENTS_MW.items():
            for (chem, mode), (soh, p_cap, p) in states.items():
                dP, L, head = np.minimum(mw, gen[act]), (gen + p)[act], (p_cap - p)[act]
                for hl in [H_LOAD, 0.2, 1.0]:
                    if hl != H_LOAD and ev != "10% trip (design)":
                        continue
                    r, _ = simulate(dP, L, head, np.full(act.sum(), BAT_MW), h_load=hl)
                    rows.append(dict(source=src, event=ev, chem=chem, operation=mode, SoH=soh, H_load=hl,
                                     hours=int(act.sum()), pct_below_49p4=100 * (r["f_min_Hz"] < F0 - UFLS_HZ).mean(),
                                     pct_at_limit=100 * r["at_limit"].mean(),
                                     f_min_median=float(np.median(r["f_min_Hz"])), f_min_worst=float(r["f_min_Hz"].min()),
                                     f_min_p1=float(np.percentile(r["f_min_Hz"], 1)),
                                     rocof0_median=float(np.median(r["rocof0_Hz_s"])),
                                     rocof500_median=float(np.median(r["rocof500_Hz_s"])),
                                     rocof500_max=float(r["rocof500_Hz_s"].max()),
                                     f_end_median=float(np.median(r["f_end_Hz"]))))
                # traces at the example hour
                i = ex
                r, tr = simulate(np.array([min(mw, gen[i])]), np.array([gen[i] + p[i]]), np.array([p_cap - p[i]]),
                                 np.array([BAT_MW]), keep=[0])
                ex_rows.append(dict(source=src, event=ev, chem=chem, operation=mode, SoH=soh, hour=str(df.t[i]),
                                    gen_MW=gen[i], dispatch_MW=p[i], load_MW=gen[i] + p[i], p_cap_MW=p_cap,
                                    spare_MW=p_cap - p[i], **{k: float(v[0]) for k, v in r.items()}))
                t = np.array(tr[0])
                ex_traces.append(pd.DataFrame({"source": src, "event": ev, "chem": chem, "operation": mode,
                                               "t": t[::5, 0], "f_Hz": t[::5, 1], "p_bat_MW": t[::5, 2]}))
        print(f"done {src}", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "re_freq_summary.csv", index=False)
    E = pd.DataFrame(ex_rows)
    E.to_csv(OUT / "re_freq_example_hours.csv", index=False)
    pd.concat(ex_traces).to_csv(OUT / "re_freq_traces.csv", index=False)

    pd.set_option("display.width", 250)
    m = R[R.H_load == H_LOAD]
    print("\n===== all producing hours, year 10 (fresh = year 0), H_load 0.5 s =====")
    print(m[["source", "event", "chem", "operation", "SoH", "pct_at_limit", "pct_below_49p4", "f_min_median",
             "f_min_p1", "f_min_worst", "rocof0_median", "rocof500_median", "rocof500_max", "f_end_median"]]
          .round(3).to_string(index=False))
    print("\n===== H_load sensitivity, design event: % hours below 49.4 Hz | worst f_min =====")
    s = R[R.event == "10% trip (design)"]
    print(s.pivot_table(index=["source", "chem", "operation"], columns="H_load",
                        values=["pct_below_49p4", "f_min_worst", "rocof500_max"]).round(3).to_string())
    print("\n===== example hours =====")
    print(E[["source", "event", "chem", "operation", "hour", "gen_MW", "dispatch_MW", "load_MW", "p_cap_MW", "spare_MW",
             "f_min_Hz", "rocof0_Hz_s", "rocof500_Hz_s", "s_below_49p4", "f_end_Hz"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
