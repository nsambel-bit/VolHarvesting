import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
T, N, sig, f = 252, 3000, 0.012, 0.5
regimes = [("Random walk", 0.0, "#2a78d6"), ("Trending", 0.2, "#eb6834"), ("Mean-reverting", -0.2, "#1baf7a")]
TXT, TXT2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

def paths(phi):
    e = rng.normal(0, sig*np.sqrt(1-phi**2), (N, T))
    l = np.zeros((N, T)); l[:, 0] = rng.normal(0, sig, N)
    for t in range(1, T): l[:, t] = phi*l[:, t-1] + e[:, t]
    return l

res = {}
for name, phi, c in regimes:
    l = paths(phi); r = np.expm1(l)
    w = np.cumsum(r, axis=1) - r
    pnl = (w*r).sum(1); ident = 0.5*(r.sum(1)**2 - (r**2).sum(1))
    x = l.sum(1); qv = (l**2).sum(1)
    logW_rb = np.log1p(f*r).sum(1); pred_rb = f*x + 0.5*f*(1-f)*qv
    logW_bh = np.log(1 - f + f*np.exp(x))
    res[name] = dict(c=c, phi=phi, e1=np.abs(pnl-ident).max(), e2=np.abs(logW_rb-pred_rb).max(),
                     trend=0.5*(x**2-qv), prem=logW_rb-logW_bh, vr=x**2/qv)

for k,v in res.items():
    print(f"{k:15s} e1={v['e1']:.1e} e2={v['e2']:.1e} VR={v['vr'].mean():.2f} trend={v['trend'].mean()*100:+.2f}% prem={v['prem'].mean()*100:+.2f}% win_trend={np.mean(v['trend']>0)*100:.0f}% win_rebal={np.mean(v['prem']>0)*100:.0f}%")
allt = np.concatenate([v['trend'] for v in res.values()]); allp = np.concatenate([v['prem'] for v in res.values()])
print(f"slope={np.polyfit(allt, allp, 1)[0]:.4f} theory={-f*(1-f):.4f} corr={np.corrcoef(allt,allp)[0,1]:.5f}")

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TXT2, "xtick.color": TXT2, "ytick.color": TXT2})
fig, ax = plt.subplots(2, 1, figsize=(6.4, 9.2), facecolor=SURF)
for a in ax: a.set_facecolor(SURF); a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)
for k,v in res.items():
    ax[0].scatter(v['trend']*100, v['prem']*100, s=9, alpha=0.5, color=v['c'], edgecolors=SURF, linewidths=0.4, label=k)
xs = np.linspace(allt.min(), allt.max(), 50)
ax[0].plot(xs*100, -f*(1-f)*xs*100, color=TXT, lw=1.2, ls="--", label="Theory: −f(1−f) × trend")
ax[0].set_xlabel("Trend-following P&L  ½(x² − QV)   [%]"); ax[0].set_ylabel("Rebalancing − buy-and-hold   [log %]")
ax[0].set_title("Rebalancing premium mirrors trend P&L", color=TXT, loc="left", fontsize=12)
ax[0].legend(fontsize=8, frameon=False, labelcolor=TXT2, markerscale=1.6)
for k,v in res.items():
    ax[1].hist(v['vr'], bins=60, range=(0,4), histtype="step", lw=2, color=v['c'], label=k)
ax[1].axvline(1, color=TXT, ls="--", lw=1)
ax[1].text(1.05, ax[1].get_ylim()[1]*0.93, "VR = 1", color=TXT2, fontsize=9)
ax[1].set_xlabel("Realized variance ratio  VR = x² / QV"); ax[1].set_ylabel("Paths")
ax[1].set_title("VR > 1: trend wins · VR < 1: rebalancing wins", color=TXT, loc="left", fontsize=12)
ax[1].legend(fontsize=8, frameon=False, labelcolor=TXT2)
plt.tight_layout(h_pad=2.5); plt.savefig("trend_vs_rebalancing.png", dpi=170, facecolor=SURF)
