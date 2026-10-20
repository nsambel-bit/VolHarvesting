"""End-to-end run: data -> residuals -> regime scores -> forecasts -> neutral portfolio -> backtest.

Usage (from the repo root, with src on PYTHONPATH or the package installed):
    python -m volharvest.run                                 # synthetic universe
    python -m volharvest.run --prices data/prices.csv --sectors data/sectors.csv
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from volharvest.config import Config
from volharvest.universe import synthetic_universe, load_csv_universe
from volharvest.factors import residualize, rolling_vol
from volharvest.regimes import all_scores
from volharvest.ensemble import Panel, walk_forward
from volharvest.backtest import run_backtest, summarize

BASELINES = ["trend_only", "reversal_only", "no_regime"]
COMBOS = ["average", "vote", "stack"]


def auc(score, label):
    s, l = score.ravel(), label.ravel().astype(bool)
    r = np.argsort(np.argsort(s)) + 1
    n1, n0 = l.sum(), (~l).sum()
    return float((r[l].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


SCENARIOS = {  # synthetic signal strength: (trend_eta, mr_kappa)
    "realistic": (0.008, 0.03),
    "strong": (0.014, 0.06),
}


def main(args):
    cfg = Config()
    cfg.universe.trend_eta, cfg.universe.mr_kappa = SCENARIOS[args.scenario]
    cfg.portfolio.trade_fraction = args.trade_fraction
    if args.cost_bps is not None:
        cfg.portfolio.cost_bps = args.cost_bps
    if args.max_weight is not None:
        cfg.portfolio.max_weight = args.max_weight
    t0 = time.time()
    U = (load_csv_universe(args.prices, args.sectors, market_col=args.market, drop_last_row=args.drop_last_row)
         if args.prices else synthetic_universe(cfg.universe))
    e, beta_hat, market = residualize(U.returns, U.sector, cfg.model.beta_window, market=U.market)
    vol = rolling_vol(e, cfg.model.vol_window)
    z = np.zeros_like(e)
    z[1:] = e[1:] / vol[:-1]                                  # standardize by yesterday's vol
    scores = all_scores(z, cfg.model.methods, cfg.model.warmup_days)
    panel = Panel(e, vol, scores, cfg.model, regime=U.regime)

    variants = BASELINES + list(cfg.model.methods) + COMBOS + (["oracle"] if U.regime is not None else [])
    results, curves, coefs, daily = {}, {}, {}, {}
    for v in variants:
        trade, alpha, ch = walk_forward(panel, v, cfg.model)
        bt = run_backtest(U.returns, panel, trade, alpha, beta_hat, U.sector, cfg)
        res = summarize(bt, market, U.returns, U.sector)
        ic = [np.corrcoef(alpha[j], panel.y[d])[0, 1] for j, d in enumerate(trade)]
        res["ic_mean"] = float(np.nanmean(ic))
        results[v] = {k: float(x) for k, x in res.items()}
        curves[v] = np.cumsum(bt["net"]).tolist()
        coefs[v] = ch[-1][1].round(4).tolist()
        daily[f"{v}__gross"], daily[f"{v}__net"] = bt["gross"], bt["net"]
        daily["day_index"] = bt["days"]

    diag = {}
    if U.regime is not None:
        trade = np.where(panel.dates >= cfg.model.warmup_days)[0]
        lab = U.regime[panel.dates[trade]]
        for k, s in {**panel.z, **{c: panel.combined[c] for c in ("average", "vote")}}.items():
            diag[k] = auc(s[trade], lab)
    zs = np.stack([panel.z[k][-260:].ravel() for k in cfg.model.methods])
    corr = np.corrcoef(zs).round(2).tolist()

    out = dict(results=results, regime_auc=diag, score_corr=dict(methods=list(cfg.model.methods), matrix=corr),
               final_coefs=coefs, runtime_s=round(time.time() - t0, 1))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "results.json").write_text(json.dumps(out, indent=2))
    np.savez(Path(args.out) / "curves.npz", **{k: np.array(c) for k, c in curves.items()})
    # dated daily P&L per variant (gross and net), for robustness analysis
    pnl = pd.DataFrame({k: v for k, v in daily.items() if k != "day_index"},
                       index=pd.DatetimeIndex(np.asarray(U.dates)[daily["day_index"]], name="date"))
    pnl.to_csv(Path(args.out) / "daily_pnl.csv")

    print(f"{'variant':14s} {'Sh gross':>8s} {'Sh net':>7s} {'IC':>6s} {'turn/wk':>7s} {'beta':>6s} {'fac R2':>6s} {'AUC':>5s}")
    for v, r in results.items():
        a = diag.get(v, float("nan"))
        print(f"{v:14s} {r['sharpe_gross']:8.2f} {r['sharpe_net']:7.2f} {r['ic_mean']:6.3f} "
              f"{r['weekly_turnover']:7.2f} {r['realized_beta']:6.3f} {r['factor_r2']:6.3f} {a:5.3f}")
    print("score correlation:", list(cfg.model.methods))
    for row in corr:
        print("  ", row)
    print(f"runtime {out['runtime_s']}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prices")
    p.add_argument("--sectors")
    p.add_argument("--out", default="output")
    p.add_argument("--scenario", choices=list(SCENARIOS), default="realistic")
    p.add_argument("--trade-fraction", type=float, default=1.0,
                   help="fraction of the gap to target traded each week (1 = full)")
    p.add_argument("--market", help="price column to use as market proxy (e.g. ^GSPC); default: equal-weighted")
    p.add_argument("--cost-bps", type=float, help="one-way cost per unit traded, overrides config")
    p.add_argument("--max-weight", type=float, help="per-name weight cap (fraction of capital), overrides config")
    p.add_argument("--drop-last-row", action="store_true", help="drop the final date (partial-day snapshot)")
    main(p.parse_args())
