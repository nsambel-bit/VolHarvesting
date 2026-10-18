"""Weekly backtest with costs, plus performance and neutrality diagnostics."""
import numpy as np

from .config import Config
from .portfolio import neutral_weights


def run_backtest(returns, panel, trade_idx, alpha_std, beta_hat, sector, cfg: Config):
    h = cfg.model.rebalance_every
    pc = cfg.portfolio
    N = returns.shape[1]
    w = np.zeros(N)
    gross, net, turnover, gross_lev, days = [], [], [], [], []
    for j, d in enumerate(trade_idx):
        t = panel.dates[d]
        vol_w = panel.vol[d] * np.sqrt(h)
        target = neutral_weights(alpha_std[j] * vol_w, vol_w ** 2, beta_hat[t], sector, pc)
        new = w + pc.trade_fraction * (target - w)
        tc = pc.cost_bps / 1e4 * np.abs(new - w).sum()
        turnover.append(np.abs(new - w).sum())
        w = new
        gross_lev.append(np.abs(w).sum())
        r = returns[t + 1: t + 1 + h] @ w
        gross.extend(r)
        rn = r.copy()
        rn[0] -= tc
        net.extend(rn)
        days.extend(range(t + 1, t + 1 + h))
    return dict(gross=np.array(gross), net=np.array(net), days=np.array(days),
                turnover=np.array(turnover), leverage=np.array(gross_lev))


def sharpe(x):
    return x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.0


def factor_check(pnl, days, market, returns, sector):
    """Realized beta to the market and R^2 of P&L on market + sector returns."""
    m = market[days]
    beta = np.cov(pnl, m)[0, 1] / m.var()
    secs = np.unique(sector)
    F = np.column_stack([m] + [returns[days][:, sector == s].mean(1) for s in secs] + [np.ones(len(days))])
    coef, *_ = np.linalg.lstsq(F, pnl, rcond=None)
    resid = pnl - F @ coef
    r2 = 1 - resid.var() / pnl.var()
    return beta, r2


def summarize(bt, market, returns, sector):
    beta, r2 = factor_check(bt["gross"], bt["days"], market, returns, sector)
    return dict(
        sharpe_gross=sharpe(bt["gross"]), sharpe_net=sharpe(bt["net"]),
        ann_ret_net=bt["net"].mean() * 252, ann_vol=bt["gross"].std() * np.sqrt(252),
        weekly_turnover=bt["turnover"].mean(), gross_leverage=bt["leverage"].mean(),
        realized_beta=beta, factor_r2=r2,
    )
