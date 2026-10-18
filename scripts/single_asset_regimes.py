import numpy as np
from scipy.special import expit, logit

SIG = 0.012; STAY = 1 - 1/120; N, T = 300, 2520; COST = 1e-4  # 1bp per unit notional traded
def simulate(phi, rng):
    z = np.zeros((N, T), int); z[:, 0] = rng.integers(0, 2, N)
    sw = rng.random((N, T)) > STAY
    for t in range(1, T): z[:, t] = np.where(sw[:, t], 1 - z[:, t-1], z[:, t-1])
    ph = np.where(z == 0, phi, -phi)            # 0 = trending, 1 = mean-reverting
    e = rng.normal(0, SIG, (N, T)); r = np.zeros((N, T)); r[:, 0] = e[:, 0]
    for t in range(1, T): r[:, t] = ph[:, t]*r[:, t-1] + e[:, t]
    return z, r

def hamilton(r, phi, stay=STAY):
    q = np.full(r.shape[0], 0.5); p = np.zeros_like(r)
    for t in range(1, r.shape[1]):
        pi = q*stay + (1-q)*(1-stay)
        l1 = np.exp(-(r[:, t]-phi*r[:, t-1])**2/(2*SIG**2)); l2 = np.exp(-(r[:, t]+phi*r[:, t-1])**2/(2*SIG**2))
        q = pi*l1/(pi*l1 + (1-pi)*l2)
        p[:, t] = q*stay + (1-q)*(1-stay)       # P(trend at t+1 | info at t)
    p[:, 0] = 0.5; return p

def rolling_ac(r, phi, n=60):
    x0, x1 = r[:, :-1], r[:, 1:]
    c = np.cumsum(np.pad(x0*x1, ((0,0),(1,0))), 1); s = np.cumsum(np.pad(x0*x0, ((0,0),(1,0))), 1)
    ac = np.zeros_like(r); ac[:, n:] = (c[:, n:] - c[:, :-n]) / (s[:, n:] - s[:, :-n])
    return expit(n*ac*2*phi)  # Gaussian likelihood ratio of AR(+phi) vs AR(-phi), ignores switching

def positions(p, r, phi, kind):
    m1, m2 = phi*r, -phi*r
    if kind == "trend": return m1/SIG**2
    if kind == "mr": return m2/SIG**2
    if kind == "static": return (0.5*m1 + 0.5*m2)/SIG**2
    if kind == "hard": return np.where(p > 0.5, m1, m2)/SIG**2
    m = p*m1 + (1-p)*m2
    if kind == "pw_novar": return m/SIG**2
    return m/(SIG**2 + p*(1-p)*(m1-m2)**2)   # "pw": full Sharpe-optimal

def sharpe(w, r):
    pnl = w[:, :-1]*r[:, 1:]
    sd = pnl.std(1); ok = sd > 0
    if not ok.any(): return 0.0, 0.0
    scale = (0.10/np.sqrt(252))/np.where(ok, sd, 1)[:, None]   # 10% vol target per path
    ws = w[:, :-1]*scale; net = ws*r[:, 1:] - COST*np.abs(np.diff(ws, axis=1, prepend=0))
    g = np.where(ok, pnl.mean(1)/np.where(ok, sd, 1)*np.sqrt(252), 0)
    n = net.mean(1)/net.std(1)*np.sqrt(252)
    return g.mean(), n.mean()

if __name__ == "__main__":
    for phi in (0.05, 0.15):
        rng = np.random.default_rng(1); z, r = simulate(phi, rng)
        zn = np.roll(z, -1, 1); truth = (zn == 0).astype(float)   # true regime at t+1
        ham = hamilton(r, phi)
        ests = {"Oracle (true regime)": truth, "Hamilton filter": ham,
                "Overconfident filter": expit(4*logit(np.clip(ham, 1e-9, 1-1e-9))),
                "Rolling 60d autocorr": rolling_ac(r, phi)}
        print(f"\n=== phi = ±{phi}  (theory oracle Sharpe ≈ {phi*np.sqrt(252):.2f}) ===")
        for k in ("trend", "mr", "static"):
            g, n = sharpe(positions(None, r, phi, k), r); print(f"  {k:28s} gross {g:5.2f}  net {n:6.2f}")
        for name, p in ests.items():
            acc = np.mean((p[:, :-1] > 0.5) == (truth[:, :-1] > 0.5))
            print(f"  -- {name}: accuracy {acc*100:.0f}%")
            for k in ("hard", "pw_novar", "pw"):
                g, n = sharpe(positions(p, r, phi, k), r); print(f"     {k:25s} gross {g:5.2f}  net {n:6.2f}")
