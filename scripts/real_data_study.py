"""Robustness study on real prices: costs, significance vs. baseline, survivorship sensitivity,
loss attribution by stock and sector.

Run from the repo root (after `pip install -e .`):
    python scripts/real_data_study.py --prices data/prices.csv --sectors data/sectors.csv --out output/real_study
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from volharvest.config import Config
from volharvest.universe import load_csv_universe, Universe
from volharvest.factors import residualize, rolling_vol
from volharvest.regimes import all_scores
from volharvest.ensemble import Panel, walk_forward
from volharvest.portfolio import neutral_weights

VARIANTS = ["trend_only", "reversal_only", "no_regime", "var_ratio", "efficiency", "autocorr",
            "hamilton", "kalman", "average", "vote", "stack"]


def build(U: Universe, cfg: Config):
    e, beta_hat, market = residualize(U.returns, U.sector, cfg.model.beta_window, market=U.market)
    vol = rolling_vol(e, cfg.model.vol_window)
    z = np.zeros_like(e)
    z[1:] = e[1:] / vol[:-1]
    scores = all_scores(z, cfg.model.methods, cfg.model.warmup_days)
    return Panel(e, vol, scores, cfg.model), beta_hat, market


def weights_path(U, panel, beta_hat, variant, cfg):
    """Held weights per rebalance (after partial trading) and the day each takes effect."""
    trade, alpha, _ = walk_forward(panel, variant, cfg.model)
    h, pc = cfg.model.rebalance_every, cfg.portfolio
    w, W, days = np.zeros(U.returns.shape[1]), [], []
    for j, d in enumerate(trade):
        t = panel.dates[d]
        v = panel.vol[d] * np.sqrt(h)
        target = neutral_weights(alpha[j] * v, v ** 2, beta_hat[t], U.sector, pc)
        w = w + pc.trade_fraction * (target - w)
        W.append(w.copy())
        days.append(t)
    return np.array(W), np.array(days)


def daily_pnl(U, W, days, h, cost_bps):
    """Daily gross and net P&L, and per-stock gross P&L, from a weight path."""
    T, N = U.returns.shape
    gross, net, idx, by_stock = [], [], [], np.zeros((0, N))
    rows = []
    prev = np.zeros(N)
    for w, t in zip(W, days):
        tc = cost_bps / 1e4 * np.abs(w - prev).sum()
        prev = w
        R = U.returns[t + 1: t + 1 + h]
        pr = R * w
        rows.append(pr)
        g = pr.sum(1)
        n = g.copy()
        n[0] -= tc
        gross.extend(g); net.extend(n); idx.extend(range(t + 1, t + 1 + len(g)))
    dates = pd.DatetimeIndex(np.asarray(U.dates)[idx])
    return (pd.Series(gross, dates), pd.Series(net, dates),
            pd.DataFrame(np.vstack(rows), index=dates, columns=U.tickers))


def sharpe(x):
    x = np.asarray(x)
    return x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.0


def block_bootstrap_sharpe_diff(a, b, n_boot=2000, block=20, seed=0):
    """Sharpe(a) - Sharpe(b) with a circular block bootstrap CI and one-sided p-value (H0: diff <= 0)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    T = len(a)
    nb = int(np.ceil(T / block))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, T, nb)
        ix = (starts[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        diffs[i] = sharpe(a[ix]) - sharpe(b[ix])
    obs = sharpe(a) - sharpe(b)
    centered = diffs - diffs.mean()
    p = float((centered >= obs).mean())
    return obs, np.percentile(diffs, [2.5, 97.5]), p


def subset(U: Universe, keep: np.ndarray) -> Universe:
    codes, names = pd.factorize(np.asarray(U.sector_names, dtype=object)[U.sector[keep]])
    return Universe(U.returns[:, keep], np.asarray(codes), U.beta_true[keep], None, U.dates,
                    [t for t, k in zip(U.tickers, keep) if k], market=U.market, sector_names=list(names))


def main(a):
    cfg = Config()
    cfg.portfolio.trade_fraction, cfg.portfolio.max_weight = a.trade_fraction, a.max_weight
    U = load_csv_universe(a.prices, a.sectors, market_col=a.market, drop_last_row=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    h = cfg.model.rebalance_every
    panel, beta_hat, market = build(U, cfg)

    # 1) all variants: weights once, P&L at several cost levels
    paths = {v: weights_path(U, panel, beta_hat, v, cfg) for v in VARIANTS}
    costs = {}
    nets5, stock_pnl = {}, {}
    for c in (0.0, 2.0, 5.0, 10.0):
        row = {}
        for v, (W, days) in paths.items():
            g, n, ps = daily_pnl(U, W, days, h, c)
            row[v] = sharpe(n)
            if c == 5.0:
                nets5[v], stock_pnl[v] = n, ps
        costs[f"{c:g}bp"] = row
    cost_tab = pd.DataFrame(costs).round(2)

    # 2) significance of each regime variant vs no_regime (net, 5bp)
    sig = {}
    for v in VARIANTS:
        if v == "no_regime":
            continue
        obs, ci, p = block_bootstrap_sharpe_diff(nets5[v].values, nets5["no_regime"].values)
        sig[v] = dict(sharpe_diff=round(obs, 3), ci_lo=round(ci[0], 3), ci_hi=round(ci[1], 3), p_one_sided=round(p, 3))
    sig_tab = pd.DataFrame(sig).T

    # 3) 2023-25 loss attribution for no_regime: by sector and top stocks
    ps = stock_pnl["no_regime"].loc["2023":]
    by_stock = ps.sum().sort_values()
    sec_name = pd.Series(np.asarray(U.sector_names, dtype=object)[U.sector], index=U.tickers)
    by_sector = ps.sum().groupby(sec_name).sum().sort_values()

    # 4) survivorship sensitivity: drop the 10 biggest full-sample compounders and rerun key variants
    total_ret = (1 + pd.DataFrame(U.returns, columns=U.tickers)).prod()
    winners = list(total_ret.sort_values(ascending=False).index[:10])
    keep = ~np.isin(U.tickers, winners)
    U2 = subset(U, keep)
    panel2, beta2, _ = build(U2, cfg)
    surv = {}
    for v in ("trend_only", "reversal_only", "no_regime", "average"):
        W, days = weights_path(U2, panel2, beta2, v, cfg)
        _, n, _ = daily_pnl(U2, W, days, h, 5.0)
        surv[v] = dict(all_88=round(sharpe(nets5[v]), 2), without_top10=round(sharpe(n), 2))
    surv_tab = pd.DataFrame(surv).T

    # save
    pd.DataFrame(nets5).to_csv(out / "daily_net_5bp.csv")
    report = dict(cost_sensitivity=cost_tab.to_dict(), significance_vs_no_regime=sig, top10_winners_removed=winners,
                  survivorship=surv_tab.to_dict(), loss_2023_25_by_sector=by_sector.round(4).to_dict(),
                  loss_2023_25_worst_stocks=by_stock.head(8).round(4).to_dict(),
                  loss_2023_25_best_stocks=by_stock.tail(5).round(4).to_dict())
    (out / "study.json").write_text(json.dumps(report, indent=2))
    print("Net Sharpe by cost level\n", cost_tab.to_string())
    print("\nSharpe difference vs no_regime (5bp, 20-day block bootstrap)\n", sig_tab.to_string())
    print("\n2023-25 no_regime P&L by sector\n", by_sector.round(4).to_string())
    print("\n2023-25 worst stocks\n", by_stock.head(8).round(4).to_string())
    print("2023-25 best stocks\n", by_stock.tail(5).round(4).to_string())
    print("\nSurvivorship: removed", winners, "\n", surv_tab.to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prices", required=True)
    p.add_argument("--sectors", required=True)
    p.add_argument("--market", default="^GSPC")
    p.add_argument("--trade-fraction", type=float, default=0.5)
    p.add_argument("--max-weight", type=float, default=0.05)
    p.add_argument("--out", default="output/real_study")
    main(p.parse_args())
