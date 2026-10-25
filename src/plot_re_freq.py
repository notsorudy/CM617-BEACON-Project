"""BEACON: frequency response figures for the solar-only / wind-only system (re_freq.py output)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "outputs", ROOT / "figures"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

T = pd.read_csv(OUT / "re_freq_traces.csv")
E = pd.read_csv(OUT / "re_freq_example_hours.csv")
LINES = [("fresh", "SoH-unaware", "k", "-", "fresh battery"),
         ("LFP", "SoH-unaware", "tab:blue", "-", "LFP year 10"),
         ("NMC", "SoH-unaware", "tab:orange", "-", "NMC year 10"),
         ("NCA", "SoH-unaware", "tab:green", "-", "NCA year 10"),
         ("NCA", "SoH-aware", "tab:green", "--", "NCA year 10, SoH-aware")]
PANELS = [("solar", "10% trip (design)"), ("wind", "10% trip (design)"), ("wind", "25% trip (stress)")]

fig, ax = plt.subplots(len(PANELS), 2, figsize=(14, 11), sharex=True)
for i, (src, ev) in enumerate(PANELS):
    e0 = E[(E.source == src) & (E.event == ev) & (E.chem == "fresh")].iloc[0]
    for chem, mode, col, ls, lab in LINES:
        x = T[(T.source == src) & (T.event == ev) & (T.chem == chem) & (T.operation == mode)]
        e = E[(E.source == src) & (E.event == ev) & (E.chem == chem) & (E.operation == mode)].iloc[0]
        ax[i, 0].plot(x.t, x.f_Hz, ls, color=col, lw=1.6, label=lab)
        ax[i, 1].plot(x.t, e.dispatch_MW + x.p_bat_MW, ls, color=col, lw=1.6,
                      label=f"{lab}: ceiling {e.p_cap_MW:.1f} MW")
        ax[i, 1].axhline(e.p_cap_MW, color=col, ls=":", lw=0.8)
    ax[i, 0].axhline(49.4, color="red", lw=0.8, ls="--")
    ax[i, 0].text(9.9, 49.42, "49.4 Hz: first load-shedding stage", ha="right", va="bottom", fontsize=8, color="red")
    ax[i, 0].set(ylabel="frequency (Hz)", xlim=(0, 10),
                 title=f"{src}, {ev}, {e0.hour[:16]} (plant {e0.gen_MW:.1f} MW)")
    ax[i, 1].set(ylabel="battery output (MW)", title="battery output (dotted: power ceiling)")
    ax[i, 0].legend(fontsize=8, loc="lower right")
    ax[i, 1].legend(fontsize=7, loc="lower right")
for a in ax[-1]:
    a.set_xlabel("time after the trip (s)")
fig.suptitle("Frequency after a sudden loss of plant output, battery as the only inertia (real hour where aged NCA falls short)")
fig.tight_layout()
fig.savefig(FIG / "re_freq_example.png", dpi=150)
plt.close(fig)
print("saved re_freq_example.png")
