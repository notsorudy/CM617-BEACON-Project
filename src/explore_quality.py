"""BEACON: quick data quality checks on the cleaned summary data (prints only)."""
import numpy as np

from load_clean import load_clean

cells, split, _ = load_clean()
print(f"{'cell':8s} {'n':>5s} {'cyc0':>5s} {'cycN':>6s} {'life':>6s} {'QD0':>6s} {'QD1':>6s} {'QDmax':>6s} "
      f"{'IR0':>7s} {'IR1':>7s} {'IRmed':>7s} {'IR=0':>5s} {'IRnan':>5s} {'QD<.5':>5s} {'QD>1.2':>6s} policy")
tot = dict(n=0, ir0=0, irnan=0)
for k, c in cells.items():
    s = c["summary"]
    ir, qd, cyc = s["IR"], s["QD"], s["cycle"]
    n = len(cyc)
    lens = {f: len(v) for f, v in s.items()}
    assert len(set(lens.values())) == 1, (k, lens)
    life = float(np.ravel(c["cycle_life"])[0])
    n_ir0 = int(np.sum(ir == 0))
    n_irnan = int(np.sum(~np.isfinite(ir)))
    tot["n"] += n; tot["ir0"] += n_ir0; tot["irnan"] += n_irnan
    print(f"{k:8s} {n:5d} {cyc[0]:5.0f} {cyc[-1]:6.0f} {life:6.0f} {qd[0]:6.3f} {qd[1]:6.3f} {np.nanmax(qd):6.3f} "
          f"{ir[0]:7.4f} {ir[1]:7.4f} {np.nanmedian(ir):7.4f} {n_ir0:5d} {n_irnan:5d} "
          f"{int(np.sum(qd < 0.5)):5d} {int(np.sum(qd > 1.2)):6d} {c['charge_policy']}")
print(tot)

# where are IR zeros (cycle index)?
for k, c in cells.items():
    ir = c["summary"]["IR"]
    z = np.where(ir == 0)[0]
    if len(z) > 1:
        print(k, "IR zero at cycle indices", z[:15], "..." if len(z) > 15 else "")
# other temperature / chargetime ranges
allv = {f: np.hstack([c["summary"][f] for c in cells.values()]) for f in ["Tavg", "Tmin", "Tmax", "chargetime", "IR", "QD"]}
for f, v in allv.items():
    print(f, "min", np.nanmin(v), "p1", np.nanpercentile(v, 1), "median", np.nanmedian(v),
          "p99", np.nanpercentile(v, 99), "max", np.nanmax(v), "nan", int(np.sum(~np.isfinite(v))))
