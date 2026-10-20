"""Data: a synthetic universe with known factor structure and per-stock regimes,
plus a loader for real price CSVs.

Return model (daily, stock i):
    r_it = beta_i * mkt_t + sec_{s(i),t} + eps_it
    eps_it = a_it + u_it,   u_it ~ N(0, sigma_i^2)

Predictable residual component a_it depends on the stock's hidden regime:
    trending:        a_it = d_it,  d_it = rho * d_i,t-1 + eta_it   (persistent drift)
    mean-reverting:  a_it = -kappa * x_i,t-1                       (reversal of recent move)
where x_it = lambda * x_i,t-1 + eps_it is an EWMA of recent residual returns.
Regimes follow independent two-state Markov chains.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .config import UniverseConfig


@dataclass
class Universe:
    returns: np.ndarray        # (T, N) total daily returns
    sector: np.ndarray         # (N,) int sector id
    beta_true: np.ndarray      # (N,)
    regime: np.ndarray | None  # (T, N) 1 = trending, 0 = mean-reverting (None for real data)
    dates: pd.DatetimeIndex
    tickers: list
    market: np.ndarray | None = None   # (T,) market proxy returns; None = equal-weighted universe
    sector_names: list | None = None


def synthetic_universe(cfg: UniverseConfig) -> Universe:
    rng = np.random.default_rng(cfg.seed)
    T, N, S = cfg.n_days, cfg.n_stocks, cfg.n_sectors
    sector = rng.permutation(np.arange(N) % S)
    beta = np.clip(rng.normal(1.0, 0.3, N), 0.3, 1.8)
    sig = cfg.idio_vol_median * np.exp(rng.normal(0, 0.35, N))

    mkt = rng.normal(cfg.mkt_drift, cfg.mkt_vol, T)
    sec = rng.normal(0, cfg.sector_vol, (T, S))

    stay = 1 - 1 / cfg.regime_days
    regime = np.zeros((T, N), dtype=np.int8)
    regime[0] = rng.integers(0, 2, N)
    flips = rng.random((T, N)) > stay
    for t in range(1, T):
        regime[t] = np.where(flips[t], 1 - regime[t - 1], regime[t - 1])

    eps = np.zeros((T, N))
    d = np.zeros(N)
    x = np.zeros(N)
    u = rng.normal(0, 1, (T, N)) * sig
    eta = rng.normal(0, 1, (T, N)) * sig * cfg.trend_eta
    for t in range(T):
        d = cfg.trend_rho * d + eta[t]
        a = np.where(regime[t] == 1, d, -cfg.mr_kappa * x)
        eps[t] = a + u[t]
        x = cfg.mr_lambda * x + eps[t]

    returns = beta[None, :] * mkt[:, None] + sec[:, sector] + eps
    dates = pd.bdate_range("2015-01-01", periods=T)
    tickers = [f"S{i:04d}" for i in range(N)]
    return Universe(returns, sector, beta, regime, dates, tickers)


def load_csv_universe(prices_csv: str, sectors_csv: str, market_col: str | None = None,
                      drop_last_row: bool = False) -> Universe:
    """Real-data entry point.

    prices_csv:  wide table, first column = date, one column per ticker (adjusted closes).
                 Columns not listed in sectors_csv (ETFs, indices) are not traded.
    sectors_csv: two columns, ticker,sector.
    market_col:  column of prices_csv to use as the market proxy (e.g. ^GSPC);
                 default None = equal-weighted average of the traded stocks.
    drop_last_row: drop the final date (e.g. when it is an intraday snapshot).
    Missing prices are forward-filled up to 5 days, then the return is set to 0.
    """
    px = pd.read_csv(prices_csv, index_col=0, parse_dates=True).sort_index()
    if drop_last_row:
        px = px.iloc[:-1]
    sec = pd.read_csv(sectors_csv).set_index("ticker")["sector"]
    tickers = [t for t in px.columns if t in sec.index]
    mkt = None
    if market_col is not None:
        mkt = px[market_col].ffill(limit=5).pct_change(fill_method=None).iloc[1:].fillna(0.0).values
    px = px[tickers].ffill(limit=5)
    rets = px.pct_change(fill_method=None).iloc[1:].fillna(0.0)
    codes, names = pd.factorize(sec.loc[tickers])
    return Universe(rets.values, np.asarray(codes), np.full(len(tickers), np.nan),
                    None, rets.index, tickers, market=mkt, sector_names=list(names))
