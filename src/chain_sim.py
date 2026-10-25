"""
BEACON: full chain, per chemistry.
  grid duty  ->  Sandia Model 1 (SoH)  ->  Severson quadratic R(SoH)  ->  P_max fraction  ->  real Grid-India hours

1. Duty (assumption): 1 equivalent full cycle per day (365 per year). Scenarios are conditions Sandia tested
   for all three chemistries: main 25 C / DoD 60 / 0.5C; harder 25 C / DoD 100 / 1C; hot 35 C / DoD 100 / 1C.
2. SoH(year) from the Sandia Model 1 random forest without 'days' (inputs: chemistry, temperature, DoD,
   discharge rate, nominal equivalent full cycles). Year 0 is SoH = 1 by definition. A pessimistic case uses
   SoH minus the chemistry's out-of-fold RMSE on unseen conditions. Points beyond the longest test of that
   condition are flagged 'beyond_test'.
3. R/R0 from the Severson quadratic (LFP); for NMC and NCA this is an assumption, so literature +30% and +100%
   curves are run too. SoH < 0.76 is flagged as R(SoH) extrapolation.
4. P_max fraction = 1 / r.
5. Grid: clean Grid-India hours from 1 Jul 2024 to 24 Jun 2025 (latest 12 months). H(t) = 5 s x synchronous
   share. Battery sized against the lowest-inertia hour in that window for a 10% loss, demand ratio
   d in {0.8, 1.0, 1.2}. Simulator and grid parameters as in power_sim.py / real_grid_sim.py.

Outputs: outputs/chain_soh.csv, outputs/chain_results.csv, figures/chain_*.png
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import power_sim as ps

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "outputs", ROOT / "figures"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
pd.set_option("display.width", 250)

EFC_PER_YEAR = 365.0
YEARS = np.arange(0, 13)
SCENARIOS = {"main 25C DoD60 0.5C": (25.0, 60.0, 0.5), "harder 25C DoD100 1C": (25.0, 100.0, 1.0),
             "hot 35C DoD100 1C": (35.0, 100.0, 1.0)}
CHEMS = ["LFP", "NMC", "NCA"]
CURVES = ["quadratic", "lit_+30%", "lit_+100%"]
DEMANDS = [0.8, 1.0, 1.2]
EVENT = 0.10
BIN = 0.005
FEATS = ["chem_LFP", "chem_NMC", "chem_NCA", "temp_C", "dod", "c_discharge", "efc_nominal"]
COL = {"LFP": "tab:blue", "NMC": "tab:orange", "NCA": "tab:green"}


def soh_table():
    model = joblib.load(OUT / "models" / "sandia_m1_nodays__forest.joblib")
    met = pd.read_csv(OUT / "sandia_m1_metrics_nodays.csv")
    rmse = met[(met.split == "conditions") & (met.algo == "forest") & met.subset.isin(CHEMS)].set_index("subset").RMSE
    pts = pd.read_csv(OUT / "sandia_soh_points.csv")
    pts["efc_nominal"] = pts.cycle_index * pts.dod / 100
    rows = []
    for sc, (t, dod, rate) in SCENARIOS.items():
        for c in CHEMS:
            tested = pts[(pts.chem == c) & (pts.temp_C == t) & (pts.dod == dod) & (pts.c_discharge == rate)]
            max_efc = tested.efc_nominal.max()
            X = pd.DataFrame({"chem_LFP": int(c == "LFP"), "chem_NMC": int(c == "NMC"), "chem_NCA": int(c == "NCA"),
                              "temp_C": t, "dod": dod, "c_discharge": rate, "efc_nominal": YEARS * EFC_PER_YEAR})
            soh = model.predict(X[FEATS])
            soh[YEARS == 0] = 1.0
            for y, s in zip(YEARS, soh):
                rows.append(dict(scenario=sc, chem=c, year=int(y), efc=y * EFC_PER_YEAR, SoH=float(s),
                                 SoH_pess=float(min(1.0, s - (rmse[c] if y > 0 else 0))), rmse=rmse[c],
                                 tested_cells=tested.cell_id.nunique(), max_tested_efc=max_efc,
                                 beyond_test=bool(y * EFC_PER_YEAR > max_efc)))
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "chain_soh.csv", index=False)
    return S


def grid_window():
    g = pd.read_csv(OUT / "grid_india_hourly_clean.csv", parse_dates=["time"])
    g = g[g.valid & (g.time >= "2024-07-01")].copy()
    g["sync"] = g.sync_share.clip(upper=1.0)
    g["sync_bin"] = (np.round(g.sync / BIN) * BIN).round(4)
    return g


def main():
    S = soh_table()
    print("Sandia tested length per scenario and chemistry (nominal equivalent full cycles, years at 365/yr):")
    t = S.groupby(["scenario", "chem"]).agg(cells=("tested_cells", "first"), max_efc=("max_tested_efc", "first"))
    t["years_covered"] = (t.max_efc / EFC_PER_YEAR).round(2)
    print(t.round(1).to_string())
    print("\npredicted SoH by year (main scenario):")
    print(S[S.scenario == "main 25C DoD60 0.5C"].pivot_table(index="year", columns="chem", values="SoH")
          .round(3).to_string())

    g = grid_window()
    print(f"\ngrid window: {g.time.min()} to {g.time.max()}, {len(g)} clean hours")
    worst = g.loc[g.sync.idxmin()]
    h_min = ps.H_SYNC * float(worst.sync_bin)
    peak_req = ps.HV / (h_min + ps.HV) * EVENT
    print(f"sizing hour {worst.time}, sync {worst.sync:.4f}, H {h_min:.3f} s, peak request {peak_req:.4f} pu")
    bins = np.sort(g.sync_bin.unique())

    # unique (r) values to simulate
    cases = []
    for _, row in S.iterrows():
        for which in ["SoH", "SoH_pess"]:
            s = row[which]
            for c in CURVES:
                cases.append(dict(scenario=row.scenario, chem=row.chem, year=row.year, soh_case=which, soh=s,
                                  curve=c, r=float(ps.r_curve(c, s)), beyond_test=row.beyond_test,
                                  r_extrap=s < ps.SOH_DATA_MIN))
    K = pd.DataFrame(cases)
    K["r_key"] = K.r.round(5)
    uniq = np.unique(np.append(K.r_key.to_numpy(), 1.0))
    print(f"\n{len(K)} chain cases, {len(uniq)} distinct r values, {len(bins)} grid bins")

    sim = []
    for d in DEMANDS:
        rr, bb = np.meshgrid(uniq, bins, indexing="ij")
        rr, bb = rr.ravel(), bb.ravel()
        n = len(rr)
        res, _ = ps.simulate(np.full(n, EVENT), np.full(n, peak_req / d), rr, np.ones(n), np.full(n, 1e9),
                             h_sys=ps.H_SYNC * bb, k_gov=bb / ps.DROOP)
        sim.append(pd.DataFrame({"design_demand": d, "r_key": np.round(rr, 5), "sync_bin": bb,
                                 "rocof": res["rocof_500ms_Hz_s"], "nadir": res["f_nadir_Hz"],
                                 "sat": res["saturated"]}))
        print(f"simulated demand ratio {d}: {n} runs", flush=True)
    SIM = pd.concat(sim, ignore_index=True)
    # energy reserve not modelled here (set very large): Task 3 and the real-grid run showed the 10 s
    # reserve never bound for inertia support.

    hours = g[["time", "sync_bin"]]
    hod = g.time.dt.hour.to_numpy()
    res_rows = []
    for d in DEMANDS:
        Sd = SIM[SIM.design_demand == d]
        fresh = Sd[Sd.r_key == 1.0].set_index("sync_bin").reindex(hours.sync_bin)
        for rk, Sr in Sd.groupby("r_key"):
            x = Sr.set_index("sync_bin").reindex(hours.sync_bin)
            dro = x.rocof.to_numpy() - fresh.rocof.to_numpy()
            dna = x.nadir.to_numpy() - fresh.nadir.to_numpy()
            matters = (np.abs(dro) > 1e-3) | (np.abs(dna) > 1e-3)
            mid = (hod >= 11) & (hod <= 13)
            res_rows.append(dict(design_demand=d, r_key=rk, pct_hours_aging_matters=100 * matters.mean(),
                                 pct_hours_saturated=100 * x.sat.to_numpy().mean(),
                                 drocof_max=np.nanmax(dro), rocof_max=x.rocof.max(),
                                 rocof_midday_mean=x.rocof.to_numpy()[mid].mean(), dnadir_min=np.nanmin(dna)))
    R = pd.DataFrame(res_rows)
    full = K.merge(R, on="r_key")
    full["pmax_frac"] = 1 / full.r
    full.to_csv(OUT / "chain_results.csv", index=False)

    q = full[(full.curve == "quadratic") & (full.soh_case == "SoH") & (full.design_demand == 1.0)]
    for sc in SCENARIOS:
        print(f"\n=== {sc}, quadratic R(SoH), demand ratio 1.0 ===")
        x = q[q.scenario == sc]
        print(x.pivot_table(index="year", columns="chem", values=["soh", "pmax_frac", "pct_hours_aging_matters",
                                                                  "drocof_max"]).round(3).to_string())
    print("\nflags: beyond Sandia test length (main scenario), first year per chemistry:")
    m = full[(full.scenario == "main 25C DoD60 0.5C") & full.beyond_test]
    print(m.groupby("chem").year.min().to_dict())
    print("\nyear 10, main scenario, demand ratio 1.0: % hours aging matters by curve and SoH case:")
    print(full[(full.scenario == "main 25C DoD60 0.5C") & (full.year == 10) & (full.design_demand == 1.0)]
          .pivot_table(index=["curve", "soh_case"], columns="chem", values="pct_hours_aging_matters")
          .round(2).to_string())
    figures(S, full)
    print("\nsaved outputs/chain_soh.csv, outputs/chain_results.csv and figures")


def figures(S, full):
    sc_list = list(SCENARIOS)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)
    for a, sc in zip(ax, sc_list):
        for c in CHEMS:
            x = S[(S.scenario == sc) & (S.chem == c)]
            a.plot(x.year, x.SoH, "o-", color=COL[c], ms=3, label=c)
            a.fill_between(x.year, x.SoH_pess, x.SoH, color=COL[c], alpha=0.15)
            b = x[x.beyond_test]
            if len(b):
                a.plot(b.year, b.SoH, "o", color="white", mec=COL[c], ms=6)
        a.axhline(ps.SOH_DATA_MIN, color="grey", ls=":", lw=1)
        a.set(title=sc, xlabel="years at 1 equivalent full cycle per day")
    ax[0].set_ylabel("predicted SoH (Sandia Model 1)")
    ax[0].legend(fontsize=8, title="open marker: beyond test length", title_fontsize=7)
    fig.suptitle("Step 2: SoH from Sandia Model 1 (band: minus out-of-fold RMSE)")
    fig.tight_layout()
    fig.savefig(FIG / "chain_soh_years.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)
    for a, sc in zip(ax, sc_list):
        for c in CHEMS:
            for curve, ls in [("quadratic", "-"), ("lit_+100%", "--")]:
                x = full[(full.scenario == sc) & (full.chem == c) & (full.curve == curve) & (full.soh_case == "SoH")
                         & (full.design_demand == 1.0)].sort_values("year")
                a.plot(x.year, x.pct_hours_aging_matters, ls=ls, color=COL[c], marker="o", ms=3,
                       label=f"{c}, {curve}")
                b = x[x.beyond_test]
                a.plot(b.year, b.pct_hours_aging_matters, "o", color="white", mec=COL[c], ms=6)
        a.set(title=sc, xlabel="years at 1 equivalent full cycle per day")
    ax[0].set_ylabel("% of hours where aging matters")
    ax[0].legend(fontsize=7, title="open marker: beyond Sandia test length\n(SoH held flat, a lower bound)",
                 title_fontsize=7)
    fig.suptitle("Steps 3-5: 10% loss, battery sized at demand ratio 1.0 for the worst hour")
    fig.tight_layout()
    fig.savefig(FIG / "chain_hours_years.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    sc = sc_list[0]
    for c in CHEMS:
        x = full[(full.scenario == sc) & (full.chem == c) & (full.curve == "quadratic") & (full.soh_case == "SoH")
                 & (full.design_demand == 1.0)].sort_values("year")
        ax[0].plot(x.year, x.pmax_frac, "o-", color=COL[c], ms=3, label=c)
        ax[1].plot(x.year, x.rocof_midday_mean, "o-", color=COL[c], ms=3, label=c)
    ax[0].set(xlabel="years", ylabel="P_max / fresh P_max (quadratic R)", title=f"Step 4: power ceiling, {sc}")
    ax[1].set(xlabel="years", ylabel="mean RoCoF 11-13 h (Hz/s)", title="Step 5: midday RoCoF, 10% loss, d = 1.0")
    for a in ax:
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "chain_pmax_rocof_years.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
