"""Why did the no-regime strategy draw down after May 2021?

Decomposes the drawdown into: costs vs. gross P&L, signal decay (IC of the trend and reversal
signals by year), hidden factor bets (market, sector, momentum, mega-cap), and stock-level
contributions with position direction.

    python scripts/drawdown_analysis.py --prices data/prices.csv --sectors data/sectors.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_data_study import build, weights_path, daily_pnl, sharpe  # noqa: E402
from volharvest.config import Config  # noqa: E402
from volharvest.universe import load_csv_universe  # noqa: E402

PEAK = "2021-05-27"
MEGA = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]


def main(a):
    cfg = Config()
    cfg.portfolio.trade_fraction, cfg.portfolio.max_weight = 0.5, 0.05
    U = load_csv_universe(a.prices, a.sectors, market_col="^GSPC", drop_last_row=True)
    panel, beta_hat, market = build(U, cfg)
    h = cfg.model.rebalance_every
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = {}

    # weights and P&L for the baseline and its two legs
    legs = {}
    for v in ("no_regime", "trend_only", "reversal_only"):
        W, days = weights_path(U, panel, beta_hat, v, cfg)
        g, n, ps = daily_pnl(U, W, days, h, 5.0)
        legs[v] = dict(W=W, days=days, gross=g, net=n, stock=ps)
    base = legs["no_regime"]
    dates = base["gross"].index
    pre, post = dates < PEAK, dates >= PEAK

    # 1) gross vs costs, per year
    yr = pd.DataFrame({"gross": base["gross"], "cost": base["gross"] - base["net"]}).groupby(dates.year).sum()
    yr["net"] = yr.gross - yr.cost
    yr = (yr * 100).round(1)
    res["by_year_pct"] = yr.to_dict()

    # 2) signal quality: weekly cross-sectional IC of trend and reversal signals, by year
    trade = np.where(panel.dates >= cfg.model.warmup_days)[0]
    ic = []
    for d in trade:
        y = panel.y[d]
        ic.append((pd.Timestamp(U.dates[panel.dates[d]]),
                   np.corrcoef(panel.trend[d], y)[0, 1], np.corrcoef(panel.rev[d], y)[0, 1]))
    ic = pd.DataFrame(ic, columns=["date", "trend_ic", "reversal_ic"]).set_index("date")
    ic_year = ic.groupby(ic.index.year).mean().round(3)
    ic_t = ic.groupby(ic.index.year).apply(lambda x: x.mean() / x.std() * np.sqrt(len(x))).round(1)
    res["ic_by_year"] = ic_year.to_dict()
    # fitted coefficients the model actually used (sign tells which way it bet)
    from volharvest.ensemble import walk_forward
    _, _, coefs = walk_forward(panel, "no_regime", cfg.model)
    coef_tab = pd.DataFrame([(pd.Timestamp(U.dates[t]).year, c[0], c[1]) for t, c in coefs],
                            columns=["refit_year", "b_trend", "b_reversal"]).set_index("refit_year").round(4)
    res["model_coefs"] = coef_tab.to_dict()

    # 3) factor bets: regress daily gross P&L on market, momentum, mega-cap factors, pre vs post
    R = pd.DataFrame(U.returns, index=pd.DatetimeIndex(U.dates), columns=U.tickers)
    px = (1 + R).cumprod()
    mom_sig = (px.shift(21) / px.shift(252) - 1)               # 12-1 month momentum, known at t-1 (shift applied below)
    rk = mom_sig.rank(axis=1, pct=True)
    mom_long, mom_short = (rk >= 2 / 3), (rk <= 1 / 3)
    mom = (R.where(mom_long.shift(1)).mean(axis=1) - R.where(mom_short.shift(1)).mean(axis=1)).fillna(0)
    mega = R[MEGA].mean(axis=1) - R.mean(axis=1)
    F = pd.DataFrame({"market": pd.Series(U.market, index=R.index), "momentum": mom, "mega_cap": mega}).loc[dates]
    fac = {}
    for name, m in (("pre_peak", pre), ("post_peak", post)):
        X = np.column_stack([F.values[m], np.ones(m.sum())])
        y = base["gross"].values[m]
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        contrib = (X[:, :3] * b[:3]).sum(0) * 100
        fac[name] = dict(beta_market=b[0], beta_momentum=b[1], beta_mega=b[2],
                         pnl_market=contrib[0], pnl_momentum=contrib[1], pnl_mega=contrib[2],
                         pnl_total=y.sum() * 100, pnl_alpha=(y.sum() - (X[:, :3] * b[:3]).sum()) * 100,
                         r2=1 - np.var(y - X @ b) / np.var(y))
    fac_tab = pd.DataFrame(fac).round(3)
    res["factor_regression"] = fac_tab.to_dict()

    # 4) stock-level contributions after the peak, with average position
    ps = base["stock"].loc[PEAK:]
    Wd = pd.DataFrame(base["W"], index=pd.DatetimeIndex(np.asarray(U.dates)[base["days"]]), columns=U.tickers)
    Wd = Wd.reindex(ps.index, method="ffill")
    stock = pd.DataFrame({"pnl_pct": ps.sum() * 100, "avg_weight_pct": Wd.mean() * 100,
                          "pct_days_short": (Wd < 0).mean() * 100,
                          "stock_return_pct": ((1 + R.loc[PEAK:, :]).prod() - 1) * 100}).round(2)
    stock = stock.sort_values("pnl_pct")
    res["worst_stocks_post_peak"] = stock.head(10).to_dict(orient="index")
    res["concentration"] = dict(total_post_pct=round(ps.sum().sum() * 100, 2),
                                worst10_pct=round(stock.pnl_pct.head(10).sum(), 2))

    # 5) opportunity set: cross-sectional dispersion of weekly residual returns, by year
    disp = pd.Series(np.nanstd(panel.y * panel.vol * np.sqrt(h), axis=1),
                     index=pd.DatetimeIndex(np.asarray(U.dates)[panel.dates]))
    res["residual_dispersion_by_year"] = (disp.groupby(disp.index.year).mean() * 100).round(2).to_dict()

    (out / "drawdown.json").write_text(json.dumps(res, indent=2, default=float))
    print("P&L by year, % of capital (5bp)\n", yr.T.to_string())
    print("\nWeekly IC by year\n", pd.concat([ic_year, ic_t.add_suffix("_tstat")], axis=1).to_string())
    print("\nModel coefficients at each yearly refit\n", coef_tab.T.to_string())
    print("\nFactor regression of daily gross P&L\n", fac_tab.to_string())
    print("\nWorst 10 stocks since peak\n", stock.head(10).to_string())
    print("\nConcentration:", res["concentration"])
    print("\nResidual weekly dispersion (%) by year\n", res["residual_dispersion_by_year"])
    for v in ("trend_only", "reversal_only"):
        print(f"{v}: net P&L since peak {legs[v]['net'].loc[PEAK:].sum()*100:.1f}%, before {legs[v]['net'].loc[:PEAK].sum()*100:.1f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prices", required=True)
    p.add_argument("--sectors", required=True)
    p.add_argument("--out", default="output/drawdown")
    main(p.parse_args())
