"""Checks that matter for a backtest: constraints hold, and no future data leaks into signals.

Run: python -m pytest -q tests  (or python tests/test_pipeline.py)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from volharvest.config import Config, PortfolioConfig
from volharvest.universe import synthetic_universe
from volharvest.factors import residualize, rolling_vol
from volharvest.regimes import all_scores
from volharvest.ensemble import Panel
from volharvest.portfolio import neutral_weights, constraint_matrix


def _pipeline(returns, sector, cfg):
    e, beta, _ = residualize(returns, sector, cfg.model.beta_window)
    vol = rolling_vol(e, cfg.model.vol_window)
    z = np.zeros_like(e)
    z[1:] = e[1:] / vol[:-1]
    scores = all_scores(z, cfg.model.methods, cfg.model.warmup_days)
    return e, beta, scores, Panel(e, vol, scores, cfg.model)


def test_neutrality_and_caps():
    rng = np.random.default_rng(0)
    N = 300
    beta, sector = rng.uniform(0.5, 1.5, N), rng.integers(0, 10, N)
    pc = PortfolioConfig()
    w = neutral_weights(rng.normal(0, 0.01, N), rng.uniform(1e-4, 4e-4, N), beta, sector, pc)
    assert np.abs(constraint_matrix(beta, sector) @ w).max() < 1e-10
    assert np.abs(w).max() <= pc.max_weight + 1e-4


def test_no_lookahead():
    cfg = Config()
    cfg.universe.n_stocks, cfg.universe.n_days = 120, 900
    U = synthetic_universe(cfg.universe)
    cut = 700                                         # after warmup, so Hamilton fit is unaffected
    shocked = U.returns.copy()
    shocked[cut + 1:] += np.random.default_rng(1).normal(0, 0.05, shocked[cut + 1:].shape)
    e0, b0, s0, p0 = _pipeline(U.returns, U.sector, cfg)
    e1, b1, s1, p1 = _pipeline(shocked, U.sector, cfg)
    assert np.allclose(e0[:cut + 1], e1[:cut + 1])
    assert np.allclose(b0[:cut + 1], b1[:cut + 1])
    for k in s0:
        assert np.allclose(s0[k][:cut + 1], s1[k][:cut + 1], equal_nan=True), k
    past = p0.dates <= cut
    assert np.allclose(p0.trend[past], p1.trend[past]) and np.allclose(p0.rev[past], p1.rev[past])
    for k in p0.z:
        assert np.allclose(p0.z[k][past], p1.z[k][past]), k


if __name__ == "__main__":
    test_neutrality_and_caps()
    test_no_lookahead()
    print("all tests passed")
