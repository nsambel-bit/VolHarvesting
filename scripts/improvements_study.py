"""Test four improvements with a development / final-test split.

Development: trading days before 2021-01-01. Final test: 2021-01-01 onward, evaluated once.
Pre-declared choices (made on development only):
  A. trade fraction in {0.5, 0.25, 0.15}                  (turnover / cost control)
  B. momentum sleeve weight in {0.33, 0.5}                 (12-1 month residual momentum)
Fixed, untuned settings:
  C. jump filter: no reversal bet for 20 days after a |z| > 4 residual day
  D. runaway guard: reversal sleeve can't fade names with |momentum z| > 2

    python scripts/improvements_study.py --prices data/prices.csv --sectors data/sectors.csv
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_data_study import daily_pnl, sharpe, block_bootstrap_sharpe_diff  # noqa: E402
from volharvest.config import Config  # noqa: E402
from volharvest.universe import load_csv_universe  # noqa: E402
from volharvest.factors import residualize, rolling_vol  # noqa: E402
from volharvest.regimes import all_scores  # noqa: E402
from volharvest.ensemble import Panel, walk_forward  # noqa: E402
from volharvest.portfolio import neutral_weights  # noqa: E402
from volharvest.enhancements import xs_z, residual_momentum, jump_mask, guard_runaway  # noqa: E402

SPLIT = "2021-01-01"
COST_BPS = 5.0


class Study:
    def __init__(self, prices, sectors):
        self.cfg = Config()
        self.cfg.portfolio.max_weight = 0.05
        U = load_csv_universe(prices, sectors, market_col="^GSPC", drop_last_row=True)
        e, beta, market = residualize(U.returns, U.sector, self.cfg.model.beta_window, market=U.market)
        vol = rolling_vol(e, self.cfg.model.vol_window)
        z = np.zeros_like(e)
        z[1:] = e[1:] / vol[:-1]
        scores = all_scores(z, self.cfg.model.methods, self.cfg.model.warmup_days)
        self.U, self.beta, self.market = U, beta, market
        self.panel = Panel(e, vol, scores, self.cfg.model)
        d = self.panel.dates
        ec = self.cfg.enhance
        self.mom = residual_momentum(e, vol, d, ec.mom_lookback, ec.mom_skip)
        self.jump = jump_mask(z, d, ec.jump_threshold, ec.jump_hold)
        self.trade = np.where(d >= self.cfg.model.warmup_days)[0]
        masked = copy.copy(self.panel)
        masked.trend = np.where(self.jump, 0.0, self.panel.trend)
        masked.rev = np.where(self.jump, 0.0, self.panel.rev)
        _, self.alpha_rev, _ = walk_forward(self.panel, "no_regime", self.cfg.model)
        _, self.alpha_rev_jump, _ = walk_forward(masked, "no_regime", self.cfg.model)

    def alphas(self, w_mom=0.0, jump=False, guard=False):
        t = self.trade
        rev = xs_z(self.alpha_rev_jump if jump else self.alpha_rev)
        if jump:
            rev = np.where(self.jump[t], 0.0, rev)
        mom_z = xs_z(self.mom[t])
        if guard:
            rev = guard_runaway(rev, mom_z, self.cfg.enhance.guard_threshold)
        return (1 - w_mom) * rev + w_mom * mom_z

    def run(self, trade_fraction=0.5, **kw):
        cfg = copy.deepcopy(self.cfg)
        cfg.portfolio.trade_fraction = trade_fraction
        h, p = cfg.model.rebalance_every, self.panel
        alpha = self.alphas(**kw)
        w, W, days = np.zeros(self.U.returns.shape[1]), [], []
        for j, d in enumerate(self.trade):
            t = p.dates[d]
            v = p.vol[d] * np.sqrt(h)
            target = neutral_weights(alpha[j] * v, v ** 2, self.beta[t], self.U.sector, cfg.portfolio)
            w = w + trade_fraction * (target - w)
            W.append(w.copy()); days.append(t)
        W, days = np.array(W), np.array(days)
        g, n, _ = daily_pnl(self.U, W, days, h, COST_BPS)
        turnover = np.abs(np.diff(W, axis=0, prepend=0)).sum(1).mean()
        return dict(gross=g, net=n, turnover=turnover, W=W, days=days)


def stats(r, start=None, end=None):
    g, n = r["gross"].loc[start:end], r["net"].loc[start:end]
    return dict(gross_sharpe=sharpe(g), net_sharpe=sharpe(n), net_ret=n.mean() * 252 * 100, vol=n.std() * np.sqrt(252) * 100)


def row(name, r):
    dev, test = stats(r, end="2020-12-31"), stats(r, start=SPLIT)
    y24 = stats(r, "2024-01-01", "2024-12-31")
    return dict(variant=name, dev_net_sharpe=dev["net_sharpe"], dev_gross_sharpe=dev["gross_sharpe"],
                test_net_sharpe=test["net_sharpe"], test_gross_sharpe=test["gross_sharpe"],
                test_net_ret_pct=test["net_ret"], net_sharpe_2024=y24["net_sharpe"], weekly_turnover=r["turnover"])


def main(a):
    S = Study(a.prices, a.sectors)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    runs, log = {}, {}

    # --- selection on development only ---
    tf_dev = {}
    for tf in (0.5, 0.25, 0.15):
        runs[f"baseline_tf{tf}"] = S.run(trade_fraction=tf)
        tf_dev[tf] = stats(runs[f"baseline_tf{tf}"], end="2020-12-31")["net_sharpe"]
    tf_star = max(tf_dev, key=tf_dev.get)
    wm_dev = {}
    for wm in (0.33, 0.5):
        r = S.run(trade_fraction=tf_star, w_mom=wm)
        runs[f"mom{wm}_tf{tf_star}"] = r
        wm_dev[wm] = stats(r, end="2020-12-31")["net_sharpe"]
    wm_star = max(wm_dev, key=wm_dev.get)
    log["selection_dev_net_sharpe"] = dict(trade_fraction=tf_dev, momentum_weight=wm_dev,
                                           chosen=dict(trade_fraction=tf_star, momentum_weight=wm_star))

    # --- each change alone (vs baseline at trade fraction 0.5), then combined ---
    base = runs["baseline_tf0.5"]
    variants = {
        "Baseline (current strategy)": base,
        "A. Slower trading (fraction 0.25)": runs["baseline_tf0.25"],
        f"B. Momentum sleeve ({wm_star:.0%})": S.run(trade_fraction=0.5, w_mom=wm_star),
        "C. Jump filter": S.run(trade_fraction=0.5, jump=True),
        "D. Runaway guard": S.run(trade_fraction=0.5, guard=True),
        "Momentum sleeve alone": S.run(trade_fraction=tf_star, w_mom=1.0),
        f"Combined (fraction {tf_star}) + B + C + D": S.run(trade_fraction=tf_star, w_mom=wm_star, jump=True, guard=True),
    }
    tab = pd.DataFrame([row(k, v) for k, v in variants.items()]).set_index("variant").round(2)

    # --- final-test significance vs baseline ---
    sig = {}
    for k, v in variants.items():
        if k.startswith("Baseline"):
            continue
        a_, b_ = v["net"].loc[SPLIT:].values, base["net"].loc[SPLIT:].values
        obs, ci, p = block_bootstrap_sharpe_diff(a_, b_)
        sig[k] = dict(diff=round(obs, 2), ci_lo=round(ci[0], 2), ci_hi=round(ci[1], 2), p_one_sided=round(p, 3))
    sig_tab = pd.DataFrame(sig).T

    # --- neutrality and factor check for the combined strategy ---
    comb = variants[f"Combined (fraction {tf_star}) + B + C + D"]
    m = pd.Series(S.market, index=pd.DatetimeIndex(S.U.dates)).loc[comb["gross"].index]
    beta_real = np.cov(comb["gross"], m)[0, 1] / m.var()
    corr_base = np.corrcoef(comb["net"], base["net"])[0, 1]
    log.update(results=tab.to_dict(orient="index"), significance_test_period=sig,
               combined_realized_beta=round(float(beta_real), 3), combined_corr_with_baseline=round(float(corr_base), 2))
    (out / "improvements.json").write_text(json.dumps(log, indent=2, default=float))
    pd.DataFrame({k: v["net"] for k, v in variants.items()}).to_csv(out / "daily_net.csv")

    print("Selection on 2007-2020 (net Sharpe):", json.dumps(log["selection_dev_net_sharpe"]))
    print("\n", tab.to_string())
    print("\nFinal test (2021-2025) net Sharpe difference vs baseline, 20-day block bootstrap\n", sig_tab.to_string())
    print(f"\nCombined: realized beta {beta_real:.3f}, correlation with baseline {corr_base:.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prices", required=True)
    p.add_argument("--sectors", required=True)
    p.add_argument("--out", default="output/improvements")
    main(p.parse_args())
