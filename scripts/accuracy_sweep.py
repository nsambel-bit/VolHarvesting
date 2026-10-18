import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.stats import norm
from single_asset_regimes import simulate, hamilton, positions, sharpe

TXT, TXT2, GRID, SURF, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6", "#eb6834"
res = {}
for phi in (0.05, 0.15):
    rng = np.random.default_rng(3); z, r = simulate(phi, rng)
    s = np.where(np.roll(z, -1, 1) == 0, 1.0, -1.0)
    orc = sharpe(positions((s > 0).astype(float), r, phi, "pw"), r)[0]
    rows = []
    for a in np.linspace(0, 3, 13):
        y = a*s + rng.normal(size=s.shape); p = expit(2*a*y)        # calibrated posterior
        acc = norm.cdf(a)
        pw = sharpe(positions(p, r, phi, "pw"), r)[0]; hd = sharpe(positions(p, r, phi, "hard"), r)[0]
        th = np.sqrt(np.mean((2*p-1)**2))
        rows.append((acc, pw/orc, hd/orc, th))
    ham = hamilton(r, phi); truth = (s > 0)
    hacc = np.mean((ham[:, :-1] > 0.5) == truth[:, :-1]); hsh = sharpe(positions(ham, r, phi, "pw"), r)[0]/orc
    res[phi] = (np.array(rows), hacc, hsh)
    print(f"phi={phi}: oracle Sharpe {orc:.2f}; Hamilton acc {hacc:.2f} -> {hsh*100:.0f}% of oracle")
    for acc, pw, hd, th in rows[::2]: print(f"  acc {acc*100:4.0f}%  prob-weighted {pw*100:5.1f}%  hard {hd*100:5.1f}%  theory √E[(2p−1)²] {th*100:5.1f}%")

rows = res[0.15][0]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TXT2, "xtick.color": TXT2, "ytick.color": TXT2})
fig, ax = plt.subplots(figsize=(6.4, 5.2), facecolor=SURF); ax.set_facecolor(SURF); ax.grid(color=GRID, lw=.6); ax.set_axisbelow(True)
ax.plot(rows[:,0]*100, rows[:,1]*100, color=BLUE, lw=2, marker="o", ms=5, mec=SURF, label="Probability-weighted (m / v)")
ax.plot(rows[:,0]*100, rows[:,2]*100, color=ORANGE, lw=2, marker="o", ms=5, mec=SURF, label="Hard switch (most likely regime)")
for phi, lab, dy in ((0.05, "Hamilton filter, weak regimes (φ=±0.05)", -16), (0.15, "Hamilton filter, strong regimes (φ=±0.15)", -20)):
    _, ha, hs = res[phi]
    ax.scatter([ha*100], [hs*100], s=70, color=TXT, zorder=5, edgecolors=SURF, linewidths=1.5)
    ax.annotate(lab, (ha*100, hs*100), xytext=(8, dy), textcoords="offset points", fontsize=8, color=TXT2)
ax.set_xlim(49, 100); ax.set_ylim(-3, 103)
ax.set_xlabel("Regime classifier accuracy (calibrated)  [%]"); ax.set_ylabel("Sharpe ratio, % of perfect-knowledge Sharpe")
ax.set_title("How good must the regime estimate be?", color=TXT, loc="left", fontsize=12)
ax.legend(fontsize=8, frameon=False, labelcolor=TXT2, loc="upper left")
plt.tight_layout(); plt.savefig("sharpe_vs_accuracy.png", dpi=170, facecolor=SURF)
